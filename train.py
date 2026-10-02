"""Training entry for the plan-level Flow Matching model.

Training uses the prepared plan tensors, a Gaussian source distribution, and
supervises continuous flow plus low-sigma discrete coordinates.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from tqdm import tqdm

from data_utils import (
    DATA_DIR,
    FLOW_ROOT,
)
from flow_solver import solve_heun
from losses import compute_flow_matching_loss
from mask_cache import load_mask_cache
from mmap_cache import load_cached
from model import (
    FlowMatchingPlanTransformer,
)
from runtime_utils import (
    augment_d4_targets,
    replace_corner_xy,
)


CKPT_DIR = (
    FLOW_ROOT / "checkpoints" / "training"
)
VIZ_DIR = FLOW_ROOT / "outputs" / "training"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 2000
BATCH = 256
LR = 1e-4
LR_END = 5e-5
WARMUP_EPOCHS = 5
SAVE_EVERY = 10
INFER_STEPS = 200
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = 1000
SEED = 42
HIDDEN = 512
HEADS = 8
HEAD_DIM = 64
NODE_LAYERS = 2
CORNER_LAYERS = 4
LOSS_VERSION = 34

TYPE_NAMES = {
    0: "living",
    1: "kitchen",
    2: "bedroom",
    3: "bathroom",
    4: "balcony",
    5: "storage",
    6: "stair",
    7: "front_door",
    8: "door",
}
TYPE_COLORS = {
    "living": "#d9d9d9",
    "kitchen": "#8da0cb",
    "bedroom": "#66c2a5",
    "bathroom": "#fc8d62",
    "balcony": "#b3b3b3",
    "storage": "#a37c52",
    "stair": "#9e9ac8",
    "front_door": "#a63603",
    "door": "#e78ac3",
}

_SHARED_DATA = None


def _as_cpu_tensor(array, dtype=None):
    return torch.from_numpy(
        np.array(array, dtype=dtype, copy=True)
    )


def time_shift(mu, sigmas):
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))


def _load_shared_data():
    global _SHARED_DATA
    if _SHARED_DATA is not None:
        return _SHARED_DATA

    clip_path = DATA_DIR / "clip" / "pooled_embeddings.npy"
    if not clip_path.exists():
        clip_path = DATA_DIR / "pooled_embeddings.npy"
    required_files = [
        DATA_DIR / "plan_norm_gt.npz",
        DATA_DIR / "topology_gt.npz",
        DATA_DIR / "masks.npz",
        DATA_DIR / "edge_mapping.npz",
        clip_path,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing required training data:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    gt = np.load(DATA_DIR / "plan_norm_gt.npz", allow_pickle=True)
    topology = np.load(DATA_DIR / "topology_gt.npz", allow_pickle=True)
    edge_mapping = np.load(DATA_DIR / "edge_mapping.npz")
    mask_cache = load_mask_cache()
    clip = np.load(clip_path, mmap_mode="r")

    plan_ids = gt["plan_ids"].astype(np.int64)
    for name, values in (
        ("topology", topology["plan_ids"]),
        ("edge_mapping", edge_mapping["plan_ids"]),
        ("masks", mask_cache["plan_ids"]),
    ):
        if not np.array_equal(plan_ids, values):
            raise ValueError(f"GT and {name} plan ids are not aligned")
    if clip.shape[0] != len(plan_ids):
        raise ValueError("CLIP embedding count does not match plan ids")
    clip_ids_path = clip_path.parent / "plan_ids.npy"
    if not clip_ids_path.is_file() or not np.array_equal(np.load(clip_ids_path), plan_ids):
        raise ValueError("CLIP plan ids are missing or misaligned; rerun encode_clip.py")

    room_corner_counts = gt["room_corner_counts"].astype(np.int16)
    room_slot_valid = (
        np.arange(28, dtype=np.int16).reshape(1, 1, 28)
        < room_corner_counts[:, :, None]
    )
    _SHARED_DATA = {
        "gt_features": load_cached("gt_features"),
        "node_features": _as_cpu_tensor(
            topology["features"],
            np.float32,
        ),
        "clip_embeddings": _as_cpu_tensor(clip, np.float32),
        "corner_room": mask_cache["corner_room"],
        "corner_conn": mask_cache["corner_conn"],
        "corner_global": mask_cache["corner_global"],
        "node_conn": mask_cache["node_conn"],
        "node_global": mask_cache["node_global"],
        "corner_next_index": _as_cpu_tensor(
            edge_mapping["corner_next_index"],
            np.int64,
        ),
        "corner_edge_valid": _as_cpu_tensor(
            edge_mapping["corner_edge_valid"],
            np.bool_,
        ),
        "room_ids": _as_cpu_tensor(gt["room_ids"], np.int64),
        "corner_ids": _as_cpu_tensor(gt["corner_ids"], np.int64),
        "ring_phase": _as_cpu_tensor(
            gt["ring_phase"],
            np.float32,
        ),
        "corner_id_norm": _as_cpu_tensor(
            gt["corner_id_norm"],
            np.float32,
        ),
        "corner_valid": _as_cpu_tensor(
            gt["corner_valid"],
            np.bool_,
        ),
        "room_corner_counts": _as_cpu_tensor(
            room_corner_counts,
            np.int16,
        ),
        "room_slot_valid": _as_cpu_tensor(
            room_slot_valid,
            np.bool_,
        ),
        "plan_ids": plan_ids,
    }
    if _SHARED_DATA["gt_features"].shape[1:] != (128, 52):
        raise ValueError("GT mmap cache has an invalid feature shape")
    if _SHARED_DATA["node_features"].shape[1:] != (20, 61):
        raise ValueError("node features have an invalid shape")
    if _SHARED_DATA["room_corner_counts"].shape != (
        len(plan_ids),
        20,
    ):
        raise ValueError("room corner counts have an invalid shape")
    if _SHARED_DATA["room_slot_valid"].shape != (
        len(plan_ids),
        20,
        28,
    ):
        raise ValueError("room slot mask has an invalid shape")
    print(
        f"loaded {len(plan_ids)} plans "
        "(gt/topology/clip/masks/edge-list/areas aligned)"
    )
    return _SHARED_DATA


class MainDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        shared = _load_shared_data()
        self.gt_features = shared["gt_features"]
        self.node_features = shared["node_features"]
        self.clip_embeddings = shared["clip_embeddings"]
        self.corner_room = shared["corner_room"]
        self.corner_conn = shared["corner_conn"]
        self.corner_global = shared["corner_global"]
        self.node_conn = shared["node_conn"]
        self.node_global = shared["node_global"]
        self.corner_next_index = shared["corner_next_index"]
        self.corner_edge_valid = shared["corner_edge_valid"]
        self.room_ids = shared["room_ids"]
        self.corner_ids = shared["corner_ids"]
        self.ring_phase = shared["ring_phase"]
        self.corner_id_norm = shared["corner_id_norm"]
        self.corner_valid = shared["corner_valid"]
        self.room_corner_counts = shared["room_corner_counts"]
        self.room_slot_valid = shared["room_slot_valid"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        corner_room = self.corner_room[index]
        corner_conn = self.corner_conn[index]
        corner_global = self.corner_global[index]
        node_conn = self.node_conn[index]
        node_global = self.node_global[index]
        corner_layout = {
            "room_ids": self.room_ids[index],
            "corner_ids": self.corner_ids[index],
            "ring_phase": self.ring_phase[index],
            "corner_id_norm": self.corner_id_norm[index],
            "corner_valid": self.corner_valid[index],
            "room_corner_counts": self.room_corner_counts[index],
            "room_slot_valid": self.room_slot_valid[index],
        }
        return (
            torch.from_numpy(
                np.array(
                    self.gt_features[index],
                    dtype=np.float32,
                    copy=True,
                )
            ),
            self.clip_embeddings[index],
            corner_room,
            corner_conn,
            corner_global,
            node_conn,
            node_global,
            self.node_features[index],
            self.corner_next_index[index],
            self.corner_edge_valid[index],
            corner_layout,
            index,
        )


def sample_sigma(
    batch_size,
    device,
    sigma1_fraction=0.08,
    sigma0_fraction=0.02,
    low_sigma_threshold=0.12,
    low_sigma_step=0.01,
):
    """Sample FM noise levels while guaranteeing endpoint coverage."""
    uniform_sigma = torch.rand(batch_size, device=device)
    time_shifted = time_shift(
        1.0,
        torch.rand(batch_size, device=device).clamp(min=1e-6),
    )
    use_uniform = torch.rand(batch_size, device=device) < 0.5
    sigma = torch.where(
        use_uniform,
        uniform_sigma,
        time_shifted,
    )
    sigma = torch.where(
        torch.rand(batch_size, device=device) < sigma1_fraction,
        torch.ones_like(sigma),
        sigma,
    )
    low_mask = sigma < low_sigma_threshold
    if low_mask.any():
        low_grid = torch.arange(
            0.0,
            low_sigma_threshold + 0.5 * low_sigma_step,
            low_sigma_step,
            device=device,
        )
        low_index = torch.randint(
            0,
            low_grid.numel(),
            (batch_size,),
            device=device,
        )
        sigma = torch.where(
            low_mask,
            low_grid[low_index],
            sigma,
        )
    if sigma0_fraction > 0.0 and batch_size > 0:
        zero_count = min(
            batch_size,
            max(1, int(round(sigma0_fraction * batch_size))),
        )
        sigma[:zero_count] = 0.0
    return sigma


def lr_at_epoch(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (
            0.1 + 0.9 * epoch / max(args.warmup_epochs, 1)
        )
    progress = (epoch - args.warmup_epochs) / max(
        1,
        args.epochs - args.warmup_epochs,
    )
    progress = min(1.0, max(0.0, progress))
    return args.lr_end + (
        args.lr - args.lr_end
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_corner_t(corner_base, xy, corner_layout):
    return replace_corner_xy(corner_base, xy, corner_layout)


def train_step(model, batch, args, train=True, fixed_sigma=None):
    (
        gt_features,
        clip_embedding,
        corner_room,
        corner_conn,
        corner_global,
        node_conn,
        node_global,
        node_features,
        corner_next_index,
        corner_edge_valid,
        corner_layout,
        _,
    ) = batch

    gt_features = gt_features.to(DEVICE)
    clip_embedding = clip_embedding.to(DEVICE)
    corner_room = corner_room.to(DEVICE)
    corner_conn = corner_conn.to(DEVICE)
    corner_global = corner_global.to(DEVICE)
    node_conn = node_conn.to(DEVICE)
    node_global = node_global.to(DEVICE)
    node_features = node_features.to(DEVICE)
    corner_next_index = corner_next_index.to(DEVICE)
    corner_edge_valid = corner_edge_valid.to(DEVICE)
    corner_layout = {
        key: value.to(DEVICE)
        for key, value in corner_layout.items()
    }

    batch_size = gt_features.shape[0]
    corner_valid = corner_layout["corner_valid"]
    if fixed_sigma is not None:
        sigma = fixed_sigma.to(DEVICE).float()
    else:
        sigma = sample_sigma(
            batch_size,
            DEVICE,
            sigma1_fraction=args.sigma1_fraction,
            sigma0_fraction=args.sigma0_fraction,
        )

    x0 = gt_features[:, :, :2].clone()
    x1 = torch.randn_like(x0)
    if train and args.d4_augment:
        (
            x0,
            corner_layout,
            corner_next_index,
            corner_edge_valid,
            corner_id_map,
        ) = augment_d4_targets(
            x0,
            corner_layout,
        )
        corner_valid = corner_layout["corner_valid"]
    xt = (
        (1.0 - sigma[:, None, None]) * x0
        + sigma[:, None, None] * x1
    )
    if train and args.perturb_scale > 0.0:
        xt = xt + torch.randn_like(xt) * (
            args.perturb_scale * sigma[:, None, None]
        )
    corner_t = make_corner_t(
        gt_features,
        xt,
        corner_layout,
    )

    with autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=args.amp and DEVICE.type == "cuda",
    ):
        velocity, discrete_logits = model(
            corner_t,
            corner_layout,
            node_features,
            None,
            clip_embedding,
            sigma,
            corner_room,
            corner_conn,
            corner_global,
            corner_next_index,
            corner_edge_valid,
            node_conn,
            node_global,
            return_discrete=True,
        )
        velocity = velocity.float()
        velocity = velocity * corner_valid.unsqueeze(-1).to(
            velocity.dtype
        )
        losses = compute_flow_matching_loss(
            velocity,
            discrete_logits,
            x0,
            x1,
            sigma,
            corner_valid,
            corner_edge_valid,
            flow_weight=args.flow_weight,
            discrete_weight=args.discrete_weight,
            discrete_threshold=args.discrete_threshold,
        )
    return {
        key: value.float() if torch.is_tensor(value) else value
        for key, value in losses.items()
    }


def reconstruct_polygons(xy, corner_layout, node_features):
    valid = corner_layout["corner_valid"]
    room_ids = corner_layout["room_ids"].long()
    corner_ids = corner_layout["corner_ids"].long()
    if valid.ndim == 2:
        if valid.shape[0] != 1:
            raise ValueError("draw_plan expects one plan at a time")
        valid = valid[0]
        room_ids = room_ids[0]
        corner_ids = corner_ids[0]
    if node_features.ndim == 3:
        node_features = node_features[0]
    type_ids = node_features[:, 20:29].argmax(dim=1)

    polygons = []
    for room_id in torch.unique(room_ids[valid]):
        room_mask = valid & (room_ids == room_id)
        indices = room_mask.nonzero(as_tuple=False).flatten()
        if indices.numel() < 3:
            continue
        order = corner_ids[indices].argsort()
        coords = xy[indices][order].detach().cpu().numpy()
        polygons.append(
            {
                "type_name": TYPE_NAMES.get(
                    int(type_ids[room_id]),
                    "?",
                ),
                "coords": coords,
            }
        )
    return polygons


def _sample_snap_threshold(args, sample):
    return float(args.discrete_snap_threshold)


def draw_plan(xy, corner_layout, node_features, axis, title):
    polygons = reconstruct_polygons(
        xy,
        corner_layout,
        node_features,
    )
    all_points = [
        point for polygon in polygons for point in polygon["coords"]
    ]
    for polygon in polygons:
        coords = polygon["coords"]
        closed = list(coords) + [coords[0]]
        xs, ys = zip(*closed)
        axis.fill(
            xs,
            ys,
            color=TYPE_COLORS.get(
                polygon["type_name"],
                "#cccccc",
            ),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )
    if all_points:
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        span_x = max(max(xs) - min(xs), 1e-6)
        span_y = max(max(ys) - min(ys), 1e-6)
        axis.set_xlim(
            min(xs) - 0.05 * span_x,
            max(xs) + 0.05 * span_x,
        )
        axis.set_ylim(
            min(ys) - 0.05 * span_y,
            max(ys) + 0.05 * span_y,
        )
    else:
        axis.set_xlim(-1.5, 1.5)
        axis.set_ylim(-1.5, 1.5)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=10)
    axis.axis("off")


@torch.no_grad()
def run_inference(
    model,
    sample,
    args,
    return_discrete=True,
):
    (
        gt_features,
        clip_embedding,
        corner_room,
        corner_conn,
        corner_global,
        node_conn,
        node_global,
        node_features,
        corner_next_index,
        corner_edge_valid,
        corner_layout,
        _,
    ) = sample

    gt_features = gt_features.unsqueeze(0).to(DEVICE)
    source_xy = torch.randn_like(gt_features[:, :, :2])
    clip_embedding = clip_embedding.unsqueeze(0).to(DEVICE)
    corner_room = corner_room.unsqueeze(0).to(DEVICE)
    corner_conn = corner_conn.unsqueeze(0).to(DEVICE)
    corner_global = corner_global.unsqueeze(0).to(DEVICE)
    node_conn = node_conn.unsqueeze(0).to(DEVICE)
    node_global = node_global.unsqueeze(0).to(DEVICE)
    node_features = node_features.unsqueeze(0).to(DEVICE)
    corner_next_index = corner_next_index.unsqueeze(0).to(DEVICE)
    corner_edge_valid = corner_edge_valid.unsqueeze(0).to(DEVICE)
    corner_layout = {
        key: value.unsqueeze(0).to(DEVICE)
        for key, value in corner_layout.items()
    }

    def forward_fn(xy, sigma):
        corner_t = make_corner_t(
            gt_features,
            xy,
            corner_layout,
        )
        model_output = model(
            corner_t,
            corner_layout,
            node_features,
            None,
            clip_embedding,
            sigma,
            corner_room,
            corner_conn,
            corner_global,
            corner_next_index,
            corner_edge_valid,
            node_conn,
            node_global,
            return_discrete=return_discrete,
        )
        if return_discrete:
            velocity, logits = model_output
        else:
            velocity = model_output
            logits = None
        return corner_t, velocity, logits

    solver_output = solve_heun(
        forward_fn,
        source_xy,
        corner_layout["corner_valid"],
        steps=args.infer_steps,
        discrete_threshold=_sample_snap_threshold(args, sample),
        discrete_refine_steps=args.infer_discrete_steps,
        discrete_update=args.infer_discrete_update,
        discrete_hard=args.infer_discrete_hard,
        low_sigma_refine=args.infer_low_sigma_discrete,
        fixed_low_sigma=args.infer_fixed_low_sigma,
        low_sigma_threshold=args.infer_low_sigma_threshold,
        low_sigma_step=args.infer_low_sigma_step,
        low_sigma_steps=args.infer_low_sigma_steps,
        final_refine=args.infer_discrete_refine,
        return_trajectory=True,
    )
    current_xy, trajectory = solver_output
    _synchronize_device()
    return (
        gt_features,
        source_xy,
        current_xy,
        node_features,
        trajectory,
        corner_layout,
    )


def _synchronize_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def save_compare(model, val_dataset, epoch, args):
    os.makedirs(VIZ_DIR, exist_ok=True)
    sample_index = (
        epoch // max(args.save_every, 1)
    ) % len(val_dataset)
    sample = val_dataset[sample_index]
    sample_plan_index = int(sample[-1])
    snap_threshold = _sample_snap_threshold(args, sample)
    print(
        f"  visualization sample: position={sample_index}, "
        f"plan={sample_plan_index}, "
        f"snap={snap_threshold:.2f}"
    )
    (
        gt_features,
        source_xy,
        output_xy,
        node_features,
        trajectory,
        corner_layout,
    ) = run_inference(model, sample, args)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_plan(
        source_xy[0],
        corner_layout,
        node_features[0],
        axes[0],
        "Noise source",
    )
    draw_plan(
        gt_features[0, :, :2],
        corner_layout,
        node_features[0],
        axes[1],
        "Ground Truth",
    )
    draw_plan(
        output_xy[0],
        corner_layout,
        node_features[0],
        axes[2],
        "Output",
    )
    figure.suptitle(
        f"Flow matching compare | epoch {epoch}",
        fontsize=14,
    )
    path = VIZ_DIR / f"train_flow_compare_{epoch:03d}.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved {path}")

    snapshot_count = min(10, len(trajectory))
    snapshot_indices = np.linspace(
        0,
        len(trajectory) - 1,
        snapshot_count,
        dtype=np.int64,
    )
    columns = 5
    rows = int(math.ceil(snapshot_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.6 * rows),
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, trajectory_index in zip(axes, snapshot_indices):
        sigma_value, snapshot_xy = trajectory[trajectory_index]
        draw_plan(
            snapshot_xy[0].to(DEVICE),
            corner_layout,
            node_features[0],
            axis,
            f"sigma={sigma_value:.4f}",
        )
    for axis in axes[snapshot_count:]:
        axis.axis("off")
    figure.suptitle(
        f"Flow matching denoise trajectory | epoch {epoch}",
        fontsize=14,
    )
    path = VIZ_DIR / f"train_flow_denoise_{epoch:03d}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved {path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lr-end", type=float, default=LR_END)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=WARMUP_EPOCHS,
    )
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--infer-steps", type=int, default=INFER_STEPS)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--infer-seed", type=int, default=20260914)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--validation-size", type=int, default=1200,
                        help="Validation plans; capped at 10 percent for small datasets")
    parser.add_argument("--val-sigma-steps", type=int, default=64)
    parser.add_argument("--val-sigma-seed", type=int, default=123)
    parser.add_argument("--val-fresh-sigma", action="store_true")
    parser.set_defaults(
        ema=True,
        ema_decay=EMA_DECAY,
        ema_warmup_steps=EMA_WARMUP_STEPS,
        hidden=HIDDEN,
        heads=HEADS,
        head_dim=HEAD_DIM,
        node_layers=NODE_LAYERS,
        corner_layers=CORNER_LAYERS,
        sigma1_fraction=0.08,
        sigma0_fraction=0.02,
        perturb_scale=0.03,
        flow_weight=1.0,
        discrete_weight=1.0,
        discrete_threshold=0.12,
        discrete_snap_threshold=0.12,
        infer_discrete_refine=False,
        infer_low_sigma_discrete=True,
        infer_discrete_update="posterior",
        infer_discrete_hard=True,
        infer_discrete_steps=12,
        infer_fixed_low_sigma=True,
        infer_low_sigma_threshold=0.12,
        infer_low_sigma_step=0.01,
        infer_low_sigma_steps=12,
        joint_bit_finetune=True,
        d4_augment=True,
        base_logits_weight=0.0,
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.infer_steps < 1:
        parser.error("--infer-steps must be positive")
    if args.infer_discrete_steps < 0:
        parser.error("--infer-discrete-steps must be non-negative")
    if args.infer_discrete_steps > args.infer_steps:
        parser.error(
            "--infer-discrete-steps cannot exceed --infer-steps"
        )
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    if args.resume and not os.path.exists(args.resume):
        parser.error(f"resume checkpoint does not exist: {args.resume}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device: {DEVICE}")

    shared = _load_shared_data()
    all_indices = np.arange(len(shared["plan_ids"]), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_indices)
    if len(all_indices) < 2 or args.validation_size < 1:
        parser.error("Training needs at least two plans and a positive validation size")
    validation_size = min(args.validation_size, max(1, len(all_indices) // 10))
    val_indices = all_indices[:validation_size]
    train_indices = all_indices[validation_size:]
    if args.train_limit > 0:
        train_indices = train_indices[: args.train_limit]
    if args.val_limit > 0:
        val_indices = val_indices[: args.val_limit]

    train_dataset = MainDataset(train_indices)
    val_dataset = MainDataset(val_indices)
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    if len(val_dataset) == 0:
        raise ValueError("validation split is empty")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=len(train_dataset) >= args.batch,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print("source sampling: gaussian noise")

    with torch.random.fork_rng():
        torch.manual_seed(args.val_sigma_seed)
        val_sigmas = sample_sigma(
            args.val_sigma_steps,
            torch.device("cpu"),
            sigma1_fraction=args.sigma1_fraction,
            sigma0_fraction=args.sigma0_fraction,
        )

    resume_checkpoint = None
    if args.resume and os.path.exists(args.resume):
        resume_checkpoint = torch.load(
            args.resume,
            map_location=DEVICE,
        )
        saved_args = resume_checkpoint.get("args", {})
        for key in (
            "hidden",
            "heads",
            "head_dim",
            "node_layers",
            "corner_layers",
        ):
            if key in saved_args:
                setattr(args, key, saved_args[key])

    model = FlowMatchingPlanTransformer(
        hidden_dimension=args.hidden,
        num_heads=args.heads,
        head_dimension=args.head_dim,
        num_node_layers=args.node_layers,
        num_corner_layers=args.corner_layers,
        detach_bit_from_velocity=not args.joint_bit_finetune,
        base_logits_weight=args.base_logits_weight,
    ).to(DEVICE)
    print(
        "model parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    ema_model = None
    if args.ema:
        if not 0.0 < args.ema_decay < 1.0:
            parser.error("--ema-decay must be in (0, 1)")
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )
    current_epoch = [0]

    def lr_lambda(_):
        epoch = current_epoch[0]
        return lr_at_epoch(epoch, args) / args.lr

    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model"])
        if ema_model is not None:
            if checkpoint.get("ema_model") is not None:
                ema_model.load_state_dict(checkpoint["ema_model"])
            else:
                ema_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(
            checkpoint.get(
                "current_epoch",
                checkpoint.get("epoch", 0),
            )
        )
        current_epoch[0] = start_epoch
        print(f"resumed from {args.resume}, start epoch {start_epoch}")

    best_val_loss = float("inf")
    state_dict = None
    if resume_checkpoint is not None:
        saved_loss_version = int(
            resume_checkpoint.get("loss_version", 0)
        )
        if saved_loss_version == LOSS_VERSION:
            best_val_loss = float(
                resume_checkpoint.get(
                    "best_val_loss",
                    float("inf"),
                )
            )
        else:
            print(
                "loss definition changed; resetting best "
                "validation loss"
            )

    ema_step = 0
    if resume_checkpoint is not None:
        ema_step = int(resume_checkpoint.get("ema_step", 0))

    for epoch in range(start_epoch, args.epochs):
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        model.train()
        running = {
            "total": 0.0,
            "flow": 0.0,
            "discrete": 0.0,
            "bit_accuracy": 0.0,
        }
        steps = 0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )
        for batch in progress_bar:
            losses = train_step(model, batch, args)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_model is not None:
                ema_step += 1
                ema_decay = min(
                    args.ema_decay,
                    (1.0 + ema_step) / (10.0 + ema_step),
                )
                with torch.no_grad():
                    for ema_parameter, parameter in zip(
                        ema_model.parameters(),
                        model.parameters(),
                    ):
                        ema_parameter.mul_(ema_decay).add_(
                            parameter.detach(),
                            alpha=1.0 - ema_decay,
                        )
                    for ema_buffer, buffer in zip(
                        ema_model.buffers(),
                        model.buffers(),
                    ):
                        ema_buffer.copy_(buffer)

            for key in running:
                running[key] += losses[key].item()
            steps += 1
            progress_bar.set_postfix(
                loss=f"{losses['total'].item():.3f}",
                flow=f"{losses['flow'].item():.2f}",
                bit=f"{losses['discrete'].item():.3f}",
                acc=f"{losses['bit_accuracy'].item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.1e}",
            )

        current_epoch[0] = epoch + 1
        lr_scheduler.step()
        average = {
            key: value / max(steps, 1)
            for key, value in running.items()
        }
        print(
            f"  Train total={average['total']:.5f} "
            f"flow={average['flow']:.5f} "
            f"bit={average['discrete']:.5f} "
            f"bit_acc={average['bit_accuracy']:.5f}"
        )
        model.eval()
        val_total = 0.0
        val_flow = 0.0
        val_parts = {
            "discrete": 0.0,
            "bit_accuracy": 0.0,
        }
        val_steps = 0
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(args.val_sigma_seed + epoch)
            if args.val_fresh_sigma:
                fresh_sigmas = sample_sigma(
                    len(val_dataset),
                    torch.device("cpu"),
                    sigma1_fraction=args.sigma1_fraction,
                    sigma0_fraction=args.sigma0_fraction,
                )
                for val_step, batch in enumerate(val_loader):
                    batch_size = batch[0].shape[0]
                    offset = val_step * batch_size
                    sigma_batch = fresh_sigmas[
                        offset : offset + batch_size
                    ].to(DEVICE)
                    losses = train_step(
                        model,
                        batch,
                        args,
                        train=False,
                        fixed_sigma=sigma_batch,
                    )
                    val_total += losses["total"].item()
                    val_flow += losses["flow"].item()
                    for key in val_parts:
                        val_parts[key] += losses[key].item()
                    val_steps += 1
            else:
                for val_step, batch in enumerate(val_loader):
                    batch_size = batch[0].shape[0]
                    offset = val_step * batch_size
                    sigma_batch = val_sigmas[
                        (torch.arange(batch_size) + offset)
                        % args.val_sigma_steps
                    ]
                    losses = train_step(
                        model,
                        batch,
                        args,
                        train=False,
                        fixed_sigma=sigma_batch,
                    )
                    val_total += losses["total"].item()
                    val_flow += losses["flow"].item()
                    for key in val_parts:
                        val_parts[key] += losses[key].item()
                    val_steps += 1
        val_loss = val_total / max(val_steps, 1)
        val_flow = val_flow / max(val_steps, 1)
        val_average = {
            key: value / max(val_steps, 1)
            for key, value in val_parts.items()
        }
        print(
            f"  Validation loss: {val_loss:.5f} "
            f"flow={val_flow:.5f} "
            f"bit={val_average['discrete']:.5f} "
            f"bit_acc={val_average['bit_accuracy']:.5f} "
            f"gap={average['flow'] - val_flow:.5f}"
        )

        os.makedirs(CKPT_DIR, exist_ok=True)
        state_dict = {
            "model": model.state_dict(),
            "ema_model": (
                ema_model.state_dict()
                if ema_model is not None
                else None
            ),
            "ema_step": ema_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "current_epoch": current_epoch[0],
            "best_val_loss": best_val_loss,
            "loss_version": LOSS_VERSION,
            "lr": optimizer.param_groups[0]["lr"],
            "args": vars(args),
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict["best_val_loss"] = best_val_loss
            torch.save(
                state_dict,
                CKPT_DIR / "flow_matching_best.pth",
            )
            print(
                f"  [New best validation loss: "
                f"{best_val_loss:.5f}]"
            )

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = CKPT_DIR / (
                f"flow_matching_epoch_{epoch + 1:03d}.pth"
            )
            torch.save(state_dict, checkpoint_path)
            _synchronize_device()
            save_compare(
                (
                    ema_model
                    if (
                        ema_model is not None
                        and ema_step >= args.ema_warmup_steps
                    )
                    else model
                ),
                val_dataset,
                epoch + 1,
                args,
            )
            _synchronize_device()
            print(f"  saved {checkpoint_path}")

    if state_dict is not None:
        torch.save(
            state_dict,
            CKPT_DIR / "flow_matching_final.pth",
        )
        _synchronize_device()
        save_compare(
            (
                ema_model
                if (
                    ema_model is not None
                    and ema_step >= args.ema_warmup_steps
                )
                else model
            ),
            val_dataset,
            current_epoch[0],
            args,
        )
        _synchronize_device()
        print("saved final compare and denoise visualizations")
    print("Training complete.")


if __name__ == "__main__":
    main()
