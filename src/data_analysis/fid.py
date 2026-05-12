import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def list_images(root):
    root = Path(root)

    files = sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ])

    if len(files) == 0:
        raise RuntimeError(f"No images found in: {root}")

    return files


def sample_files(files, n, seed):
    if n > len(files):
        raise ValueError(f"Requested {n} images, but only found {len(files)}.")

    rng = random.Random(seed)
    return rng.sample(files, n)


def sample_two_disjoint_sets(files, n, seed):
    """
    Creates COCO-A and COCO-B with no overlapping images.
    Useful for the real-vs-real baseline:
        FID(COCO-A, COCO-B)
    """
    if 2 * n > len(files):
        raise ValueError(
            f"Need at least {2 * n} images for two disjoint sets, "
            f"but only found {len(files)}."
        )

    rng = random.Random(seed)
    selected = rng.sample(files, 2 * n)

    return selected[:n], selected[n:]


class ImageFileDataset(Dataset):
    def __init__(self, files, image_size=299):
        self.files = list(files)

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),  # float tensor in [0, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as img:
            img = img.convert("RGB")
        return self.transform(img)


def make_loader_from_files(files, batch_size=64, num_workers=2):
    dataset = ImageFileDataset(files)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def compute_inception_score_from_files(
    files,
    batch_size=64,
    splits=10,
    device=None,
    num_workers=2,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    loader = make_loader_from_files(
        files,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    metric = InceptionScore(
        splits=splits,
        normalize=True,
    ).to(device)

    for imgs in tqdm(loader, desc="IS"):
        imgs = imgs.to(device)
        metric.update(imgs)

    mean, std = metric.compute()

    return mean.item(), std.item()


@torch.no_grad()
def compute_fid_kid_from_files(
    real_files,
    fake_files,
    batch_size=64,
    device=None,
    num_workers=2,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    real_loader = make_loader_from_files(
        real_files,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    fake_loader = make_loader_from_files(
        fake_files,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
    ).to(device)

    kid = KernelInceptionDistance(
        feature=2048,
        subsets=50,
        subset_size=min(1000, len(real_files), len(fake_files)),
        normalize=True,
    ).to(device)

    for imgs in tqdm(real_loader, desc="FID/KID real"):
        imgs = imgs.to(device)
        fid.update(imgs, real=True)
        kid.update(imgs, real=True)

    for imgs in tqdm(fake_loader, desc="FID/KID fake"):
        imgs = imgs.to(device)
        fid.update(imgs, real=False)
        kid.update(imgs, real=False)

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()

    return {
        "fid": fid_score,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
    }


def mean_std(values):
    values = torch.tensor(values, dtype=torch.float64)

    mean = values.mean().item()

    if len(values) > 1:
        std = values.std(unbiased=True).item()
    else:
        std = 0.0

    return mean, std
    
N = 1503
BATCH_SIZE = 64
ROUNDS = 50

vilstrub_dir = "../../../data/ViLStrUB/images/total"
coco_dir = "../../../data/coco2017/test2017"

vilstrub_all = list_images(vilstrub_dir)
coco_all = list_images(coco_dir)

print(f"ViLStrUB images found: {len(vilstrub_all)}")
print(f"COCO images found: {len(coco_all)}")

# Use all ViLStrUB images if it has exactly 1503.
# Otherwise, sample 1503 once.
if len(vilstrub_all) == N:
    vilstrub_files = vilstrub_all
else:
    vilstrub_files = sample_files(vilstrub_all, N, seed=999)

print("\nComputing ViLStrUB Inception Score once...")
is_vilstrub_mean, is_vilstrub_split_std = compute_inception_score_from_files(
    vilstrub_files,
    batch_size=BATCH_SIZE,
    splits=10,
)

print(
    f"IS ViLStrUB: "
    f"{is_vilstrub_mean:.4f} ± {is_vilstrub_split_std:.4f} "
    f"(TorchMetrics split std)"
)


is_coco_runs = []

fid_vilstrub_coco_runs = []
kid_vilstrub_coco_runs = []

fid_coco_real_real_runs = []
kid_coco_real_real_runs = []


for r in range(ROUNDS):
    print(f"\n================ Round {r + 1}/{ROUNDS} ================")

    # COCO-A and COCO-B are disjoint random subsets of 1503 images.
    coco_a_files, coco_b_files = sample_two_disjoint_sets(
        coco_all,
        N,
        seed=10_000 + r,
    )

    # 1. IS for COCO subset
    is_coco_mean, is_coco_split_std = compute_inception_score_from_files(
        coco_a_files,
        batch_size=BATCH_SIZE,
        splits=10,
    )

    is_coco_runs.append(is_coco_mean)

    # 2. ViLStrUB vs COCO-A
    metrics_vilstrub_coco = compute_fid_kid_from_files(
        real_files=coco_a_files,
        fake_files=vilstrub_files,
        batch_size=BATCH_SIZE,
    )

    fid_vilstrub_coco_runs.append(metrics_vilstrub_coco["fid"])
    kid_vilstrub_coco_runs.append(metrics_vilstrub_coco["kid_mean"])

    # 3. COCO-A vs COCO-B real-vs-real baseline
    metrics_coco_real_real = compute_fid_kid_from_files(
        real_files=coco_a_files,
        fake_files=coco_b_files,
        batch_size=BATCH_SIZE,
    )

    fid_coco_real_real_runs.append(metrics_coco_real_real["fid"])
    kid_coco_real_real_runs.append(metrics_coco_real_real["kid_mean"])

    print(f"Round {r + 1} IS COCO: {is_coco_mean:.4f}")
    print(f"Round {r + 1} FID ViLStrUB vs COCO: {metrics_vilstrub_coco['fid']:.4f}")
    print(f"Round {r + 1} KID ViLStrUB vs COCO: {metrics_vilstrub_coco['kid_mean']:.6f}")
    print(f"Round {r + 1} FID COCO-A vs COCO-B: {metrics_coco_real_real['fid']:.4f}")
    print(f"Round {r + 1} KID COCO-A vs COCO-B: {metrics_coco_real_real['kid_mean']:.6f}")


# Final summary
is_coco_mean, is_coco_std = mean_std(is_coco_runs)

fid_vc_mean, fid_vc_std = mean_std(fid_vilstrub_coco_runs)
kid_vc_mean, kid_vc_std = mean_std(kid_vilstrub_coco_runs)

fid_cc_mean, fid_cc_std = mean_std(fid_coco_real_real_runs)
kid_cc_mean, kid_cc_std = mean_std(kid_coco_real_real_runs)

print("\n================ FINAL RESULTS ================")

print(
    f"IS ViLStrUB: "
    f"{is_vilstrub_mean:.4f} ± {is_vilstrub_split_std:.4f} "
    f"(split std)"
)

print(
    f"IS COCO-1503: "
    f"{is_coco_mean:.4f} ± {is_coco_std:.4f} "
    f"(across {ROUNDS} random COCO subsets)"
)

print(
    f"FID ViLStrUB vs COCO-1503: "
    f"{fid_vc_mean:.4f} ± {fid_vc_std:.4f}"
)

print(
    f"KID ViLStrUB vs COCO-1503: "
    f"{kid_vc_mean:.6f} ± {kid_vc_std:.6f}"
)

print(
    f"FID COCO-1503-A vs COCO-1503-B: "
    f"{fid_cc_mean:.4f} ± {fid_cc_std:.4f}"
)

print(
    f"KID COCO-1503-A vs COCO-1503-B: "
    f"{kid_cc_mean:.6f} ± {kid_cc_std:.6f}"
)