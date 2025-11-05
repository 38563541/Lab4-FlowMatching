"""
Evaluate DDPM Teacher Model (Task 4)

This script evaluates a trained DDPM teacher model by:
1. Generating samples from the model
2. Optionally computing FID score against real data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.ddpm_teacher.model import DiffusionModule
from image_common.ddpm_teacher.scheduler import DDPMScheduler


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    # Load DDPM model
    print(f"Loading DDPM checkpoint from {args.ckpt_path}")
    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ckpt_path)
    ddpm.eval()
    ddpm = ddpm.to(device)

    # Setup scheduler for inference
    T = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        T,
        beta_1=1e-4,
        beta_T=0.02,
        mode='linear',
    ).to(device)

    print(f"Image resolution: {ddpm.image_resolution}")
    print(f"Use CFG: {ddpm.network.use_cfg}")
    if ddpm.network.use_cfg:
        print(f"Guidance scale: {args.guidance_scale}")

    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    print(f"Generating {total_num_samples} samples...")

    for i in tqdm(range(num_batches), desc="Generating samples"):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if ddpm.network.use_cfg:
            # Sample balanced class labels (1, 2, 3 for cat, dog, wild)
            # Generate equal numbers of each class
            labels = []
            num_classes = 3
            for c in range(1, num_classes + 1):
                labels.extend([c] * (B // num_classes))
            # Add remaining
            remaining = B - len(labels)
            labels.extend([1] * remaining)
            class_label = torch.tensor(labels).to(device)

            samples = ddpm.sample(
                B,
                class_label=class_label,
                guidance_scale=args.guidance_scale
            )
        else:
            samples = ddpm.sample(B)

        pil_images = tensor_to_pil_image(samples)

        for j, img in zip(range(sidx, eidx), pil_images):
            img.save(save_dir / f"{j}.png")

    print(f"\nGenerated {total_num_samples} samples to {save_dir}")

    # Optionally compute FID
    if args.compute_fid:
        print("\nComputing FID score...")
        print(f"Run: python -m fid.measure_fid {args.real_data_path} {save_dir}")

        if args.real_data_path:
            try:
                from fid.measure_fid import calculate_fid_given_paths
                paths = [args.real_data_path, str(save_dir)]
                fid_value = calculate_fid_given_paths(paths, img_size=256, batch_size=64)
                print(f"FID: {fid_value:.2f}")

                # Save FID to file
                with open(save_dir / "fid_score.txt", "w") as f:
                    f.write(f"FID: {fid_value:.2f}\n")
            except Exception as e:
                print(f"Failed to compute FID: {e}")
                print("You can compute FID manually using:")
                print(f"python -m fid.measure_fid {args.real_data_path} {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DDPM Teacher Model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to DDPM checkpoint")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance scale (if model uses CFG)")
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID score")
    parser.add_argument("--real_data_path", type=str, default="data/afhq/val",
                        help="Path to real data for FID computation")

    args = parser.parse_args()
    main(args)
