"""
Phase 1: Generate 2-Rectified Flow Training Data from DDPM Teacher

This script generates synthetic training pairs (x_0, x_1) by sampling from a pretrained DDPM model.
These pairs are used to train a 2-Rectified Flow model with straighter generation paths.

The DDPM teacher uses Classifier-Free Guidance (CFG) with guidance scale α₁ (default: 7.5)
to generate high-quality target images.

Reference: Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.ddpm_teacher.teacher import DDPMTeacher


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save configuration
    config = vars(args)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = f"cuda:{args.gpu}"

    # Load DDPM teacher model from Assignment 1
    print(f"Loading DDPM teacher checkpoint from {args.ddpm_ckpt_path}")
    ddpm_teacher = DDPMTeacher(args.ddpm_ckpt_path, device)
    ddpm_teacher.model.eval()

    if args.use_cfg:
        assert ddpm_teacher.model.network.use_cfg, "The DDPM model was not trained with CFG support."
        num_classes = ddpm_teacher.model.network.num_classes
        print(f"Using CFG with {num_classes} classes (including null class)")
    else:
        num_classes = None

    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    ddpm_steps = ddpm_teacher.model.var_scheduler.num_train_timesteps
    print(f"\nGenerating {total_num_samples} training pairs for 2-Rectified Flow...")
    print(f"DDPM teacher uses {ddpm_steps} denoising steps")
    print(f"CFG guidance scale (α₁): {args.cfg_scale}")
    print(f"This is Phase 1: Creating straight-path teacher model\n")

    for i in tqdm(range(num_batches), desc="Generating 2-RF training data"):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        # Sample initial noise x_0
        shape = (B, 3, ddpm_teacher.image_resolution, ddpm_teacher.image_resolution)
        x_0 = torch.randn(shape).to(device)

        # Sample class labels if using CFG
        if args.use_cfg:
            labels = torch.randint(1, num_classes, (B,)).to(device)
        else:
            labels = None

        # Generate x_1 using DDPM teacher with CFG (guidance scale α₁)
        x_1 = ddpm_teacher.sample(
            shape,
            class_label=labels,
            guidance_scale=args.cfg_scale,
        )

        # Save the pairs to disk
        for j in range(B):
            sample_idx = sidx + j

            torch.save(x_0[j].cpu(), save_dir / f"{sample_idx:06d}_x0.pt")
            torch.save(x_1[j].cpu(), save_dir / f"{sample_idx:06d}_x1.pt")

            if args.use_cfg:
                torch.save(labels[j].cpu(), save_dir / f"{sample_idx:06d}_label.pt")

            # Save first 100 samples as images for inspection
            if args.save_images and sample_idx < 100:
                img_x1 = tensor_to_pil_image(x_1[j].cpu(), single_image=True)
                img_x1.save(save_dir / f"{sample_idx:06d}_x1.png")

    print(f"\n2-Rectified Flow training data saved to {save_dir}")
    print(f"Total samples: {total_num_samples}")

    # Save metadata
    metadata = {
        "num_samples": total_num_samples,
        "teacher_type": "ddpm",
        "teacher_inference_steps": ddpm_steps,
        "use_cfg": args.use_cfg,
        "cfg_scale": args.cfg_scale,
        "image_resolution": ddpm_teacher.image_resolution,
        "num_classes": num_classes,
        "phase": "phase1_2rf_training",
        "purpose": "Train 2-Rectified Flow with straight paths from DDPM",
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("Phase 1 data generation complete!")
    print(f"Next step: Train 2-Rectified Flow model with:")
    print(f"python -m task4_instaflow.phase1_train_2rf \\")
    print(f"  --reflow_data_path {save_dir} \\")
    if args.use_cfg:
        print(f"  --use_cfg \\")
    print(f"  --train_num_steps 100000")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate 2-Rectified Flow training data from DDPM teacher"
    )
    parser.add_argument("--ddpm_ckpt_path", type=str, required=True,
                        help="Path to DDPM teacher checkpoint from Assignment 1 (.ckpt file)")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of training pairs to generate (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for generation")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--save_dir", type=str, default="data/afhq_2rf",
                        help="Directory to save 2-RF training dataset")
    parser.add_argument("--use_cfg", action="store_true",
                        help="Use classifier-free guidance (required if DDPM is conditional)")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="CFG guidance scale α₁ for DDPM teacher (default: 7.5)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save first 100 samples as images for inspection")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
