"""
Generate InstaFlow Distillation Dataset (Task 4)

This script generates training pairs (x_0, x_1) for distilling a Flow Matching model
into a one-step generator. Unlike multi-step ODE integration, InstaFlow learns to map
noise directly to data in a single forward pass.

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
from image_common.fm import FlowMatching
from image_common.ddpm_teacher.teacher import DDPMTeacher

def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save configuration
    config = vars(args)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = f"cuda:{args.gpu}"

    # Load teacher model
    if args.teacher_type == 'fm':
        print(f"Loading FM teacher checkpoint from {args.ckpt_path}")
        teacher = FlowMatching(None, None)
        teacher.load(args.ckpt_path)
    elif args.teacher_type == 'ddpm':
        print(f"Loading DDPM teacher checkpoint from {args.ddpm_ckpt_path}")
        teacher = DDPMTeacher(args.ddpm_ckpt_path, device)
    else:
        raise ValueError(f"Unknown teacher type: {args.teacher_type}")
    
    teacher.model.eval()
    teacher.model.to(device)

    if args.use_cfg:
        assert teacher.model.network.use_cfg, "The model was not trained with CFG support."
        num_classes = teacher.model.network.num_classes
        print(f"Using CFG with {num_classes} classes (including null class)")

    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    print(f"Generating {total_num_samples} distillation pairs for InstaFlow...")
    if args.teacher_type == 'fm':
        print(f"Using {args.num_inference_steps} ODE steps for teacher model sampling")

    for i in tqdm(range(num_batches), desc="Generating InstaFlow dataset"):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        # Sample initial noise
        shape = (B, 3, teacher.image_resolution, teacher.image_resolution)
        x_0 = torch.randn(shape).to(device)

        # Sample class labels if using CFG
        if args.use_cfg:
            labels = torch.randint(1, num_classes, (B,)).to(device)
        else:
            labels = None

        # Generate x_1 using teacher model
        x_1 = teacher.sample(
            shape,
            class_label=labels,
            guidance_scale=args.cfg_scale,
            num_inference_timesteps=args.num_inference_steps
        )

        # Save the pairs to disk
        for j in range(B):
            sample_idx = sidx + j

            torch.save(x_0[j].cpu(), save_dir / f"{sample_idx:06d}_x0.pt")
            torch.save(x_1[j].cpu(), save_dir / f"{sample_idx:06d}_x1.pt")

            if args.use_cfg:
                torch.save(labels[j].cpu(), save_dir / f"{sample_idx:06d}_label.pt")

            if args.save_images and sample_idx < 100:
                img_x1 = tensor_to_pil_image(x_1[j].cpu(), single_image=True)
                img_x1.save(save_dir / f"{sample_idx:06d}_x1.png")

    print(f"InstaFlow distillation dataset saved to {save_dir}")
    print(f"Total samples: {total_num_samples}")

    metadata = {
        "num_samples": total_num_samples,
        "teacher_type": args.teacher_type,
        "teacher_inference_steps": args.num_inference_steps if args.teacher_type == 'fm' else teacher.model.var_scheduler.num_train_timesteps,
        "use_cfg": args.use_cfg,
        "teacher_cfg_scale": args.cfg_scale if args.use_cfg else None,
        "image_resolution": teacher.image_resolution,
        "num_classes": num_classes if args.use_cfg else None,
        "distillation_type": "instaflow",
        "target_steps": 1,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("Dataset generation complete!")
    print(f"Next step: Train InstaFlow one-step generator with:")
    print(f"python -m task4_instaflow.train_instaflow")
    print(f"  --distill_data_path {save_dir}")
    if args.use_cfg:
        print(f"  --use_cfg")
    print(f"  --train_num_steps 100000")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distillation dataset for InstaFlow one-step generation")
    parser.add_argument("--teacher_type", type=str, default='fm', choices=['fm', 'ddpm'], help="Type of teacher model to use.")
    parser.add_argument("--ckpt_path", type=str, help="Path to teacher model (base FM or rectified flow)")
    parser.add_argument("--ddpm_ckpt_path", type=str, help="Path to DDPM teacher model checkpoint")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of distillation pairs to generate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--save_dir", type=str, default="data/afhq_instaflow", help="Directory to save distillation dataset")
    parser.add_argument("--use_cfg", action="store_true", help="Use classifier-free guidance")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG guidance scale for teacher model")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of ODE steps for FM teacher generation (20-50 recommended)")
    parser.add_argument("--save_images", action="store_true", help="Save first 100 samples as images for inspection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args() # This line has a typo: parse_ should be parse

    if args.teacher_type == 'ddpm' and not args.ddpm_ckpt_path:
        parser.error("--ddpm_ckpt_path is required when --teacher_type is 'ddpm'")
    if args.teacher_type == 'fm' and not args.ckpt_path:
        parser.error("--ckpt_path is required when --teacher_type is 'fm'")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)