
"""Group 06: cranial MRI enhancement and quality assessment.

The Colab notebook embeds this module, so uploading the notebook alone is enough.
"""
# %% imports
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm


@dataclass
class Config:
    # "quick" verifies the workflow; "standard" is the project experiment.
    profile: str = "standard"
    dataset_id: str = "ds000221"
    n_subjects: int = 20
    image_size: int = 256  # Preserve more spatial detail than the old 128x128 setting.
    slices_per_subject: int = 48
    epochs: int = 25
    qa_epochs: int = 12
    batch_size: int = 4  # Smaller batches keep 256x256 U-Net training practical on Colab GPUs.
    base_channels: int = 24
    learning_rate: float = 1e-3
    patience: int = 7
    seed: int = 42
    num_workers: int = 0
    noise_kind: str = "mixed"  # gaussian, rician, or mixed
    root: str = "/content/mri_group06"
    local_nifti_dir: str = ""  # optional BIDS NIfTI data instead of download
    resume: bool = True

    @classmethod
    def for_profile(cls, profile="standard", root="/content/mri_group06"):
        cfg = cls(profile=profile, root=str(root))
        if profile == "quick":
            cfg.n_subjects, cfg.slices_per_subject = 10, 16
            cfg.image_size, cfg.batch_size = 128, 8
            cfg.epochs, cfg.qa_epochs, cfg.base_channels = 3, 3, 16
        elif profile != "standard":
            raise ValueError("Choose 'quick' or 'standard'.")
        return cfg


def setup(cfg):
    assert cfg.image_size >= 32 and cfg.image_size % 8 == 0
    assert cfg.n_subjects >= 10 and cfg.n_subjects % 10 == 0, "Use 10, 20, 30, ... subjects for an exact 70/30 split."
    assert cfg.noise_kind in ("gaussian", "rician", "mixed")
    assert cfg.epochs > 0 and cfg.qa_epochs > 0
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    print("Device:", device, "| PyTorch:", torch.__version__)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CPU detected. Select a GPU in Colab for the standard experiment.")
    return device


# %% download
S3 = "https://s3.amazonaws.com/openneuro.org"
S3NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def http_session():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def list_s3_entries(session, prefix, delimiter=None):
    objects, prefixes, token = [], [], None
    for _ in range(1000):
        params = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        response = session.get(S3, params=params, timeout=(15, 120))
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        prefixes.extend(item.findtext("s3:Prefix", namespaces=S3NS)
                        for item in tree.findall("s3:CommonPrefixes", S3NS))
        for obj in tree.findall("s3:Contents", S3NS):
            key = obj.findtext("s3:Key", namespaces=S3NS)
            if key:
                objects.append({"key": key, "size": int(obj.findtext("s3:Size", namespaces=S3NS)),
                                "etag": obj.findtext("s3:ETag", namespaces=S3NS)})
        if tree.findtext("s3:IsTruncated", namespaces=S3NS) != "true":
            break
        token = tree.findtext("s3:NextContinuationToken", namespaces=S3NS)
        if not token:
            raise RuntimeError("Incomplete S3 listing: missing continuation token.")
    else:
        raise RuntimeError("Dataset listing exceeded safety limit.")
    return objects, sorted(prefixes)


def list_t1_objects(dataset_id, subject_limit=20, seed=42):
    """Sample subject folders first, avoiding a costly listing of all derivatives."""
    if not re.fullmatch(r"ds\d{6}", dataset_id):
        raise ValueError("Expected an OpenNeuro ID such as ds000221.")
    session = http_session()
    _, subject_prefixes = list_s3_entries(session, dataset_id + "/sub-", delimiter="/")
    objects, found = [], 0
    for prefix in np.random.default_rng(seed).permutation(subject_prefixes):
        entries, _ = list_s3_entries(session, str(prefix))
        matches = [row for row in entries if row["key"].endswith(("_T1w.nii.gz", "_T1w.nii"))
                   and "/anat/" in row["key"] and "/derivatives/" not in row["key"]]
        if matches:
            for row in matches:
                row["subject"] = re.search(r"/(sub-[^/]+)/", row["key"]).group(1)
            objects.extend(matches)
            found += 1
        if found >= subject_limit:
            break
    if not objects:
        raise RuntimeError("No T1w NIfTI scans found. Check dataset access or set local_nifti_dir.")
    return sorted(objects, key=lambda row: row["key"])


def download_file(session, key, destination, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        return destination
    temporary = destination.with_name(destination.name + ".part")
    url = S3 + "/" + quote(key, safe="/")
    with session.get(url, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise IOError(f"Incomplete download: {key}; rerun this cell to retry.")
    temporary.replace(destination)
    return destination


def acquire_dataset(cfg):
    data_dir = Path(cfg.root) / "data" / cfg.dataset_id
    data_dir.mkdir(parents=True, exist_ok=True)
    if cfg.local_nifti_dir:
        candidates = []
        for file in sorted(Path(cfg.local_nifti_dir).rglob("*")):
            if file.name.endswith(("_T1w.nii.gz", "_T1w.nii")) and "derivatives" not in file.parts:
                subject = next((p for p in file.parts if p.startswith("sub-")), None)
                if subject:
                    candidates.append({"subject": subject, "key": str(file.resolve()),
                                       "size": file.stat().st_size, "etag": "local", "path": str(file)})
    else:
        listing_path = data_dir / f"t1_listing_seed{cfg.seed}_n{cfg.n_subjects}.json"
        if listing_path.exists():
            candidates = json.loads(listing_path.read_text())
        else:
            candidates = list_t1_objects(cfg.dataset_id, cfg.n_subjects, cfg.seed)
            listing_path.write_text(json.dumps(candidates, indent=2))
    # Select exactly one acquisition per person; never split repeated scans across sets.
    per_subject = {}
    for row in sorted(candidates, key=lambda row: row["key"]):
        per_subject.setdefault(row["subject"], row)
    subjects = np.array(sorted(per_subject))
    if len(subjects) < cfg.n_subjects:
        raise ValueError(f"Need {cfg.n_subjects} distinct subjects; found {len(subjects)}.")
    subjects = np.random.default_rng(cfg.seed).permutation(subjects)[:cfg.n_subjects]
    records, session = [], http_session()
    for subject in tqdm(subjects, desc="Download T1 MRI subjects"):
        record = dict(per_subject[subject])
        if not cfg.local_nifti_dir:
            filename = Path(record["key"]).name
            file = download_file(session, record["key"], data_dir / subject / filename, record["size"])
            record["path"] = str(file)
        # Force a header read, catching HTML/error payloads and broken downloads now.
        image = nib.load(record["path"])
        if len(image.shape) != 3:
            raise ValueError(f"Expected a 3D anatomical MRI: {record['path']} {image.shape}")
        records.append(record)
    if not cfg.local_nifti_dir:
        for name in ("dataset_description.json", "README", "CHANGES", "participants.tsv"):
            try:
                download_file(session, cfg.dataset_id + "/" + name, data_dir / name)
            except requests.HTTPError as error:
                if error.response.status_code != 404:
                    raise
    print(f"Using {len(records)} subjects from {cfg.dataset_id}; source files stay outside result ZIP.")
    return records


# %% preprocessing
def normalize_volume(volume):
    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0, posinf=0, neginf=0)
    positive = volume[volume > 0]
    if positive.size < 100:
        raise ValueError("Volume has too few positive voxels for T1 intensity normalization.")
    low, high = np.percentile(positive, [1, 99])
    if high <= low:
        raise ValueError("MRI intensity range is constant.")
    return np.clip((volume - low) / (high - low), 0, 1).astype(np.float32)


def resize_with_padding(image, image_size):
    """Resize a grayscale slice without warping its aspect ratio; pad with MRI background."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not image.size:
        raise ValueError("Expected a nonempty 2D grayscale MRI slice.")
    height, width = image.shape
    scale = min(image_size / height, image_size / width)
    new_h = max(1, min(image_size, int(round(height * scale))))
    new_w = max(1, min(image_size, int(round(width * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    top, left = (image_size - new_h) // 2, (image_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return np.clip(canvas, 0, 1)


def split_subjects(records, cfg):
    # Input order was randomized at acquisition. Keep this split in the manifest.
    n_development = len(records) * 7 // 10
    development, test = records[:n_development], records[n_development:]
    n_validation = max(1, int(round(len(development) * 0.2)))
    splits = {"train": development[:-n_validation], "validation": development[-n_validation:], "test": test}
    ids = {name: {row["subject"] for row in rows} for name, rows in splits.items()}
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
    assert len(development) / len(records) == 0.7
    print("70% development (training + internal validation), 30% held-out test:")
    print({name: len(rows) for name, rows in splits.items()})
    return splits


def prepare_data(records, cfg):
    splits = split_subjects(records, cfg)
    # Isolate incompatible experiments instead of silently reusing their checkpoints.
    signature = {key: value for key, value in asdict(cfg).items()
                 if key not in ("root", "resume", "epochs", "qa_epochs", "num_workers", "local_nifti_dir")}
    signature["subjects"] = {name: [{k: row[k] for k in ("subject", "key", "size", "etag")} for row in rows]
                             for name, rows in splits.items()}
    run_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "subject_manifest.json").write_text(json.dumps(splits, indent=2))
    (run_dir / "experiment_signature.json").write_text(json.dumps(signature, indent=2))
    data = {}
    for split, rows in splits.items():
        images, metadata = [], []
        for row in tqdm(rows, desc=f"Prepare {split}"):
            image = nib.as_closest_canonical(nib.load(row["path"]))
            volume = normalize_volume(image.get_fdata(dtype=np.float32))
            # Canonical NIfTI: axis 2 is the superior/inferior (axial) axis.
            depth = volume.shape[2]
            valid = [z for z in range(int(depth * 0.15), int(depth * 0.85))
                     if np.mean(volume[:, :, z] > 0.08) >= 0.12]
            if len(valid) < cfg.slices_per_subject:
                raise ValueError(f"Too few nonempty axial slices in {row['subject']}: {len(valid)}. Reduce slices_per_subject.")
            chosen = np.array(valid)[np.linspace(0, len(valid) - 1, cfg.slices_per_subject).round().astype(int)]
            assert len(set(chosen)) == cfg.slices_per_subject
            for z in chosen:
                # Transpose/flip is only for consistent radiological display, applied to every pair.
                axial = np.flipud(volume[:, :, z].T)
                resized = resize_with_padding(axial, cfg.image_size)
                images.append(np.ascontiguousarray(resized, dtype=np.float32))
                metadata.append({"subject": row["subject"], "slice": int(z), "source": row["key"]})
        data[split] = {"images": np.stack(images), "metadata": metadata}
    print("Slices:", {name: len(values["images"]) for name, values in data.items()})
    print("Result folder:", run_dir)
    return data, run_dir


SCENARIOS = {"mild": (2, 0.03), "moderate": (3, 0.06), "severe": (4, 0.10)}


def degrade(clean, rng, noise_kind="rician", scale=None, sigma=None):
    scale = int(rng.choice([2, 3, 4])) if scale is None else int(scale)
    sigma = float(rng.uniform(0.02, 0.12)) if sigma is None else float(sigma)
    kind = str(rng.choice(["gaussian", "rician"])) if noise_kind == "mixed" else noise_kind
    h, w = clean.shape
    small = cv2.resize(clean, (max(8, w // scale), max(8, h // scale)), interpolation=cv2.INTER_AREA)
    blurred = np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
    n1 = rng.normal(0, sigma, clean.shape)
    if kind == "rician":
        n2 = rng.normal(0, sigma, clean.shape)
        degraded = np.sqrt((blurred + n1) ** 2 + n2 ** 2)
    elif kind == "gaussian":
        degraded = blurred + n1
    else:
        raise ValueError("Noise must be gaussian, rician, or mixed.")
    return np.clip(degraded, 0, 1).astype(np.float32)


class MRIPairs(Dataset):
    def __init__(self, images, cfg, training=False, seed_offset=0):
        self.images, self.cfg, self.training = images, cfg, training
        self.seed_offset, self.epoch = seed_offset, 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        seed = self.cfg.seed + self.seed_offset + index + (self.epoch * 100003 if self.training else 0)
        rng = np.random.default_rng(seed)
        clean = self.images[index].copy()
        if self.training:
            clean = np.rot90(clean, int(rng.integers(4)))
            if rng.random() < 0.5:
                clean = np.fliplr(clean)
        bad = degrade(clean, rng, self.cfg.noise_kind)
        return torch.from_numpy(bad[None]), torch.from_numpy(np.ascontiguousarray(clean[None]))


def make_pair_loaders(data, cfg):
    datasets = {name: MRIPairs(data[name]["images"], cfg, name == "train", 0 if name == "train" else 200000)
                for name in ("train", "validation")}
    # Bump the signature when preprocessing or the enhancement objective changes, so
    # an older checkpoint cannot silently resume under the new training setup.
    for ds in datasets.values():
        ds.input_signature = f"paired-mri-v2-{cfg.image_size}-{cfg.noise_kind}-edge-loss"
    return {name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=name == "train",
                             num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
            for name, ds in datasets.items()}


# %% models
class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
                                    nn.GroupNorm(4, outgoing), nn.ReLU(inplace=True),
                                    nn.Conv2d(outgoing, outgoing, 3, padding=1),
                                    nn.GroupNorm(4, outgoing), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(1, base), ConvBlock(base, base*2), ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(base*4, base*8)
        self.up3, self.up2, self.up1 = (nn.ConvTranspose2d(base*8, base*4, 2, stride=2),
                                       nn.ConvTranspose2d(base*4, base*2, 2, stride=2),
                                       nn.ConvTranspose2d(base*2, base, 2, stride=2))
        self.d3, self.d2, self.d1 = ConvBlock(base*8, base*4), ConvBlock(base*4, base*2), ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
        # Start with the input image and learn a bounded correction.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        middle = self.bridge(self.pool(c))
        z = self.d3(torch.cat([self.up3(middle), c], dim=1))
        z = self.d2(torch.cat([self.up2(z), b], dim=1))
        z = self.d1(torch.cat([self.up1(z), a], dim=1))
        return (x + 0.5 * torch.tanh(self.out(z))).clamp(0, 1)


def differentiable_ssim(x, y):
    # Gaussian 11x11 SSIM, data_range=1. Evaluation uses scikit-image's Wang settings.
    coord = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-(coord ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    blur = lambda value: torch.nn.functional.conv2d(value, kernel)
    mx, my = blur(x), blur(y)
    vx = (blur(x*x) - mx*mx).clamp_min(0)
    vy = (blur(y*y) - my*my).clamp_min(0)
    covariance = blur(x*y) - mx*my
    value = ((2*mx*my + 0.01**2) * (2*covariance + 0.03**2)) / (
        (mx*mx + my*my + 0.01**2) * (vx + vy + 0.03**2))
    return value.mean()


def enhancement_loss(prediction, target):
    """Balance pixel fidelity, structure, and edge retention to limit over-smoothing."""
    pixel = nn.functional.l1_loss(prediction, target)
    structure = 1 - differentiable_ssim(prediction, target)
    pred_dx, target_dx = prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1]
    pred_dy, target_dy = prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    edges = 0.5 * (nn.functional.l1_loss(pred_dx, target_dx) + nn.functional.l1_loss(pred_dy, target_dy))
    return 0.60 * pixel + 0.30 * structure + 0.10 * edges


class QualityNet(nn.Module):
    """No-reference, synthetic-domain PSNR/SSIM regressor; not a clinical quality classifier."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*4*4, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.features(x))


def fit_model(model, loaders, device, cfg, run_dir, name, epochs, loss_fn):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    best, start, stale, history = float("inf"), 0, 0, []
    last_path, best_path = Path(run_dir) / f"{name}_last.pt", Path(run_dir) / f"{name}_best.pt"
    input_signature = getattr(loaders["train"].dataset, "input_signature", "paired-mri")
    if cfg.resume and last_path.exists():
        # Only load this project's own checkpoint files. Never load untrusted pickle checkpoints.
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("input_signature") == input_signature:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start, best, stale = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["stale"]
            history = checkpoint["history"]
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
            print(f"Resuming {name} at epoch {start + 1}")
        else:
            print(f"{name}: training inputs changed; starting a fresh model instead of resuming stale weights.")
    for epoch in range(start, epochs):
        if stale >= cfg.patience:
            print(f"{name}: early stopping already reached; loading best validation checkpoint.")
            break
        started = time.perf_counter()
        losses = {}
        for phase in ("train", "validation"):
            training = phase == "train"
            model.train(training)
            if hasattr(loaders[phase].dataset, "epoch"):
                loaders[phase].dataset.epoch = epoch
            total, count = 0.0, 0
            for inputs, targets in loaders[phase]:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    predictions = model(inputs)
                    loss = loss_fn(predictions, targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite {name} loss.")
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                total += loss.item() * len(inputs)
                count += len(inputs)
            losses[phase] = total / count
        if device.type == "cuda":
            torch.cuda.synchronize()
        improved = losses["validation"] < best - 1e-6
        if improved:
            best, stale = losses["validation"], 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, best_path)
        else:
            stale += 1
        row = {"epoch": epoch + 1, "train_loss": losses["train"], "validation_loss": losses["validation"],
               "seconds": time.perf_counter() - started}
        history.append(row)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "best": best, "stale": stale, "history": history, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "input_signature": input_signature}, last_path)
        pd.DataFrame(history).to_csv(Path(run_dir) / f"{name}_history.csv", index=False)
        print(f"{name} {epoch+1:02d}/{epochs}: train={losses['train']:.5f}, val={losses['validation']:.5f}, {row['seconds']:.1f}s")
    if not best_path.exists():
        raise RuntimeError("No trained checkpoint available. Set epochs to at least 1.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    model.eval()
    return model, pd.DataFrame(history)


def plot_history(history, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.epoch, history.train_loss, label="Training")
    ax.plot(history.epoch, history.validation_loss, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Loss", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


# %% quality
def true_metrics(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Metrics require aligned 2D arrays of identical shape.")
    reference, candidate = np.asarray(reference, np.float32), np.asarray(candidate, np.float32)
    psnr = peak_signal_noise_ratio(reference, candidate, data_range=1.0)
    ssim = structural_similarity(reference, candidate, data_range=1.0, gaussian_weights=True,
                                 sigma=1.5, use_sample_covariance=False)
    # Foreground is intensity-derived head tissue; this is not an anatomical brain segmentation.
    mask = reference > 0.08
    mse = float(np.mean((reference[mask] - candidate[mask]) ** 2)) if mask.any() else float("nan")
    foreground = float(-10 * np.log10(mse)) if mse > 0 else (float("inf") if mse == 0 else float("nan"))
    return {"psnr_db": float(psnr), "ssim": float(ssim), "foreground_psnr_db": foreground}


@torch.inference_mode()
def enhance_array(image, model, device):
    tensor = torch.from_numpy(np.ascontiguousarray(image[None, None], dtype=np.float32)).to(device)
    return model(tensor)[0, 0].cpu().numpy()


def make_qa_dataset(split_data, enhancer, device, cfg, seed_offset):
    inputs, targets = [], []
    for index, clean in enumerate(tqdm(split_data["images"], desc="Prepare quality-estimator labels")):
        rng = np.random.default_rng(cfg.seed + seed_offset + index)
        degraded = degrade(clean, rng, cfg.noise_kind)
        enhanced = enhance_array(degraded, enhancer, device)
        for candidate in (degraded, enhanced):
            metrics = true_metrics(clean, candidate)
            inputs.append(candidate[None])
            targets.append([np.clip(metrics["psnr_db"], 0, 60) / 60,
                            (np.clip(metrics["ssim"], -1, 1) + 1) / 2])
    return torch.utils.data.TensorDataset(torch.from_numpy(np.stack(inputs).astype(np.float32)),
                                          torch.tensor(targets, dtype=torch.float32))


def make_qa_loaders(data, enhancer, device, cfg):
    signature = hashlib.sha256()
    for name, value in enhancer.state_dict().items():
        signature.update(name.encode())
        signature.update(value.detach().cpu().numpy().tobytes())
    loaders = {}
    for name, offset in (("train", 400000), ("validation", 500000)):
        dataset = make_qa_dataset(data[name], enhancer, device, cfg, offset)
        dataset.input_signature = signature.hexdigest()
        loaders[name] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=name == "train", num_workers=0)
    return loaders


@torch.inference_mode()
def estimate_quality(image, qa_model, device):
    x = torch.from_numpy(np.ascontiguousarray(image[None, None], dtype=np.float32)).to(device)
    result = qa_model(x)[0].cpu().numpy()
    return {"estimated_psnr_db": float(result[0] * 60), "estimated_ssim": float(result[1] * 2 - 1)}


# %% evaluation
def timed_call(callable_fn, device=None):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def evaluate(data, enhancer, qa_model, device, cfg, run_dir):
    # Fixed unseen subjects and fixed degradation seeds. No tuning on these results.
    rows, qa_rows, examples = [], [], {}
    kinds = ("gaussian", "rician") if cfg.noise_kind == "mixed" else (cfg.noise_kind,)
    dummy = data["test"]["images"][0]
    for _ in range(3):
        enhance_array(dummy, enhancer, device)
        estimate_quality(dummy, qa_model, device)
    chosen_indices = [len(data["test"]["images"]) // 2]
    for kind_index, kind in enumerate(kinds):
        for severity_index, (severity, (scale, sigma)) in enumerate(SCENARIOS.items()):
            for index, clean in enumerate(tqdm(data["test"]["images"], desc=f"Test {kind}/{severity}")):
                rng = np.random.default_rng(cfg.seed + 700000 + kind_index*100000 + severity_index*10000 + index)
                bad = degrade(clean, rng, kind, scale, sigma)
                smooth, smooth_ms = timed_call(lambda: cv2.GaussianBlur(bad, (3, 3), 0.7))
                restored, model_ms = timed_call(lambda: enhance_array(bad, enhancer, device), device)
                outputs = {"Before (degraded)": (bad, 0.0), "Gaussian baseline": (smooth, smooth_ms),
                           "After (U-Net)": (restored, model_ms)}
                common = {**data["test"]["metadata"][index], "noise": kind, "severity": severity,
                          "scale": scale, "sigma": sigma}
                for method, (candidate, milliseconds) in outputs.items():
                    metrics = true_metrics(clean, candidate)
                    rows.append({**common, "method": method, **metrics, "processing_ms": milliseconds})
                    if method != "Gaussian baseline":
                        estimates, qa_ms = timed_call(lambda: estimate_quality(candidate, qa_model, device), device)
                        qa_rows.append({**common, "method": method, **metrics, **estimates,
                                        "qa_processing_ms": qa_ms})
                if index in chosen_indices and kind == kinds[-1]:
                    examples[severity] = {"reference": clean, "before": bad, "baseline": smooth,
                                          "after": restored, "noise": kind, "metadata": common}
    results, quality = pd.DataFrame(rows), pd.DataFrame(qa_rows)
    results.to_csv(Path(run_dir) / "per_slice_metrics.csv", index=False)
    quality.to_csv(Path(run_dir) / "quality_predictions.csv", index=False)
    summary = results.groupby(["noise", "severity", "method"], sort=False).agg(
        n=("ssim", "size"), psnr_mean_db=("psnr_db", "mean"), psnr_std_db=("psnr_db", "std"),
        ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
        foreground_psnr_mean_db=("foreground_psnr_db", "mean"),
        mean_ms=("processing_ms", "mean"), median_ms=("processing_ms", "median"),
        p95_ms=("processing_ms", lambda x: float(np.percentile(x, 95)))).reset_index()
    summary.to_csv(Path(run_dir) / "evaluation_summary.csv", index=False)
    subjects = results.groupby(["subject", "noise", "severity", "method"], sort=False)[["psnr_db", "ssim"]].mean().reset_index()
    subjects.to_csv(Path(run_dir) / "subject_metrics.csv", index=False)
    paired = subjects.pivot(index=["subject", "noise", "severity"], columns="method", values=["psnr_db", "ssim"])
    gains = pd.DataFrame({"psnr_gain_db": paired[("psnr_db", "After (U-Net)")] - paired[("psnr_db", "Before (degraded)")],
                          "ssim_gain": paired[("ssim", "After (U-Net)")] - paired[("ssim", "Before (degraded)")]})
    gains.reset_index().to_csv(Path(run_dir) / "before_after_subject_gains.csv", index=False)
    qa_report = []
    for method, group in quality.groupby("method"):
        for metric, estimated in (("psnr_db", "estimated_psnr_db"), ("ssim", "estimated_ssim")):
            error = group[estimated] - group[metric]
            qa_report.append({"method": method, "metric": metric, "mae": float(error.abs().mean()),
                              "rmse": float(np.sqrt(np.mean(error**2))),
                              "spearman": float(group[estimated].corr(group[metric], method="spearman"))})
    qa_report = pd.DataFrame(qa_report)
    qa_report.to_csv(Path(run_dir) / "quality_estimator_accuracy.csv", index=False)
    mean_gains = gains.mean().to_dict()
    if mean_gains["psnr_gain_db"] <= 0 or mean_gains["ssim_gain"] <= 0:
        print("The model did not improve both average metrics in this run. Report the measured results; quick runs are not final experiments.")
    print("Mean subject-level before/after gains:", mean_gains)
    return results, summary, quality, qa_report, examples


def plot_comparisons(examples, run_dir):
    fig, axes = plt.subplots(len(examples), 5, figsize=(17, 3.5*len(examples)), squeeze=False)
    for row, (severity, example) in enumerate(examples.items()):
        panels = [("Reference MRI", example["reference"]), ("Before: simulated low quality", example["before"]),
                  ("Gaussian baseline", example["baseline"]), ("After: U-Net", example["after"]),
                  ("Absolute U-Net error", np.abs(example["after"] - example["reference"]))]
        for column, (label, image) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(image, cmap="magma" if column == 4 else "gray", vmin=0, vmax=0.25 if column == 4 else 1)
            title = label
            if column == 0:
                title += f"\n{example['noise']} / {severity}"
            if 0 < column < 4:
                values = true_metrics(example["reference"], image)
                title += f"\nPSNR {values['psnr_db']:.2f} dB | SSIM {values['ssim']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        for name in ("reference", "before", "baseline", "after"):
            cv2.imwrite(str(Path(run_dir) / f"example_{severity}_{name}.png"), (example[name]*255).round().astype(np.uint8))
    fig.suptitle("Fixed held-out example: same subject and slice across severity levels", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "before_after_comparison.png", dpi=160)
    plt.show()
    plt.close(fig)


def plot_metric_summary(results, run_dir):
    import inspect
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    order = ["Before (degraded)", "Gaussian baseline", "After (U-Net)"]
    for ax, metric, title in zip(axes, ["psnr_db", "ssim", "processing_ms"], ["PSNR (dB)", "SSIM", "Processing time (ms/image)"]):
        groups = [results.loc[results.method == method, metric].values for method in order]
        labels = ["Before", "Gaussian", "U-Net"]
        keyword = "tick_labels" if "tick_labels" in inspect.signature(ax.boxplot).parameters else "labels"
        ax.boxplot(groups, **{keyword: labels}, showfliers=False)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "metrics_and_processing_time.png", dpi=150)
    plt.show()
    plt.close(fig)


def plot_quality_accuracy(quality, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, actual, estimated, label in zip(axes, ["psnr_db", "ssim"], ["estimated_psnr_db", "estimated_ssim"], ["PSNR (dB)", "SSIM"]):
        ax.scatter(quality[actual], quality[estimated], s=6, alpha=0.25)
        low = min(quality[actual].min(), quality[estimated].min())
        high = max(quality[actual].max(), quality[estimated].max())
        ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
        ax.set(xlabel=f"Measured {label}", ylabel=f"Estimated {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "quality_estimator_accuracy.png", dpi=150)
    plt.show()
    plt.close(fig)


# %% inference
def quality_band(estimated_ssim, medium=0.60, good=0.80):
    """Project bands on estimated SSIM; these are not radiologist quality labels."""
    if not 0 <= medium < good <= 1:
        raise ValueError("Thresholds must satisfy 0 <= medium < good <= 1.")
    score = float(estimated_ssim)
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError("Estimated SSIM must be finite and between -1 and 1.")
    return "Good" if score >= good else "Medium" if score >= medium else "Bad"


def label_quality_report(report, medium=0.60, good=0.80):
    """Add before/after quality labels without mutating the original report."""
    labeled = dict(report)
    for key in ("before_estimated", "after_estimated"):
        estimates = dict(report[key])
        estimates["quality_label"] = quality_band(estimates["estimated_ssim"], medium, good)
        labeled[key] = estimates
    labeled["quality_bands"] = {"basis": "Estimated SSIM", "medium_at_or_above": medium,
                                "good_at_or_above": good,
                                "scope": "Editable project thresholds; not validated clinical quality classes."}
    return labeled


class MRIPipeline:
    """Reusable product module: enhancement + estimated quality + optional true metrics."""
    def __init__(self, enhancer, quality_model, device):
        self.enhancer = enhancer.eval().to(device)
        self.quality_model = quality_model.eval().to(device)
        self.device = device

    @classmethod
    def from_folder(cls, folder, device=None):
        folder = Path(folder)
        cfg = Config(**json.loads((folder / "config.json").read_text()))
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        enhancer, quality_model = UNet(cfg.base_channels), QualityNet()
        enhancer.load_state_dict(torch.load(folder / "enhancer_best.pt", map_location="cpu", weights_only=False)["model"])
        quality_model.load_state_dict(torch.load(folder / "quality_best.pt", map_location="cpu", weights_only=False)["model"])
        instance = cls(enhancer, quality_model, device)
        instance.image_size = cfg.image_size
        return instance

    def process(self, normalized_image, reference=None):
        image = np.asarray(normalized_image, dtype=np.float32)
        if image.ndim != 2 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Supply one finite, normalized grayscale image in [0,1].")
        if min(image.shape) < 32 or any(side % 8 for side in image.shape):
            raise ValueError("Image dimensions must be at least 32 and divisible by 8.")
        if hasattr(self, "image_size") and image.shape != (self.image_size, self.image_size):
            raise ValueError(f"This model was trained at {self.image_size}x{self.image_size}; resize the input first.")
        def operation():
            before = estimate_quality(image, self.quality_model, self.device)
            enhanced = enhance_array(image, self.enhancer, self.device)
            after = estimate_quality(enhanced, self.quality_model, self.device)
            return enhanced, before, after
        (enhanced, before, after), elapsed_ms = timed_call(operation, self.device)
        report = {"before_estimated": before, "after_estimated": after, "pipeline_ms": elapsed_ms,
                  "score_scope": "Synthetic Gaussian/Rician degradation of T1 MRI; estimates require external validation."}
        if reference is not None:
            report["before_measured"] = true_metrics(reference, image)
            report["after_measured"] = true_metrics(reference, enhanced)
        return enhanced, label_quality_report(report)


def load_demo_image(file, image_size):
    file = Path(file)
    if file.name.lower().endswith((".nii", ".nii.gz")):
        image = nib.as_closest_canonical(nib.load(file))
        if len(image.shape) != 3:
            raise ValueError("Upload a 3D T1 NIfTI volume.")
        volume = normalize_volume(image.get_fdata(dtype=np.float32))
        array = np.flipud(volume[:, :, volume.shape[2] // 2].T)
    else:
        array = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise ValueError("Use PNG, JPG, BMP, TIFF, .nii or .nii.gz.")
        array = array.astype(np.float32) / 255
    # Keep the full slice shape ratio; square stretching can make anatomy look distorted.
    result = resize_with_padding(array, image_size).astype(np.float32)
    if not np.isfinite(result).all() or float(np.ptp(result)) < 1e-6:
        raise ValueError("The image is empty or constant. Upload a visible brain MRI slice.")
    return result

