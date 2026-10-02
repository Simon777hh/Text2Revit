"""Inference-only runtime. Does not load datasets or import the trainer."""
from types import SimpleNamespace
import os
import torch
from flow_solver import solve_heun
from runtime_utils import replace_corner_xy

DEVICE = torch.device("cpu" if os.environ.get("TEXT2REVIT_DEVICE") == "cpu"
                      else "cuda" if torch.cuda.is_available() else "cpu")

def build_parser():
    # Preserve the production trainer's solver settings without argparse imports.
    class Settings:
        def parse_args(self, argv):
            return SimpleNamespace(infer_steps=int(argv[1]), infer_discrete_steps=12,
                infer_low_sigma_discrete=True, infer_fixed_low_sigma=True,
                infer_discrete_update="posterior", infer_discrete_hard=True,
                infer_low_sigma_threshold=0.12, infer_low_sigma_step=0.01,
                infer_low_sigma_steps=12, infer_discrete_refine=False,
                discrete_snap_threshold=0.12)
    return Settings()

@torch.no_grad()
def run_inference(model, sample, args, return_discrete=True):
    (base, clip, room_mask, connection_mask, global_mask, node_connection,
     node_global, nodes, next_index, edge_valid, layout, _) = sample
    base = base.unsqueeze(0).to(DEVICE)
    source = torch.randn_like(base[:, :, :2])
    clip, room_mask, connection_mask, global_mask, nodes, next_index, edge_valid, node_connection, node_global = [
        value.unsqueeze(0).to(DEVICE) for value in
        (clip, room_mask, connection_mask, global_mask, nodes, next_index,
         edge_valid, node_connection, node_global)]
    layout = {key: value.unsqueeze(0).to(DEVICE) for key, value in layout.items()}
    def forward(xy, sigma):
        corners = replace_corner_xy(base, xy, layout)
        result = model(corners, layout, nodes, None, clip, sigma, room_mask,
            connection_mask, global_mask, next_index, edge_valid,
            node_connection, node_global, return_discrete=return_discrete)
        velocity, logits = result if return_discrete else (result, None)
        return corners, velocity, logits
    result, trajectory = solve_heun(forward, source, layout["corner_valid"],
        steps=args.infer_steps, discrete_threshold=args.discrete_snap_threshold,
        discrete_refine_steps=args.infer_discrete_steps,
        discrete_update=args.infer_discrete_update,
        discrete_hard=args.infer_discrete_hard,
        low_sigma_refine=args.infer_low_sigma_discrete,
        fixed_low_sigma=args.infer_fixed_low_sigma,
        low_sigma_threshold=args.infer_low_sigma_threshold,
        low_sigma_step=args.infer_low_sigma_step,
        low_sigma_steps=args.infer_low_sigma_steps,
        final_refine=args.infer_discrete_refine, return_trajectory=True)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return base, source, result, nodes, trajectory, layout
