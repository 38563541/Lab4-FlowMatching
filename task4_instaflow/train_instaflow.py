"""
Training Script for InstaFlow One-Step Generator (Task 4)

Trains a one-step generator via distillation from a pretrained Flow Matching model.
The model learns to generate high-quality images in a single forward pass.

Reference: Liu et al., "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"

Usage:
    python image_fm_todo/train_instaflow.py \
        --distill_data_path data/afhq_instaflow \
        --use_cfg \
        --train_num_steps 100000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dotmap import DotMap
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import tensor_to_pil_image
from image_common.fm import FlowMatching, FMScheduler
from image_common.network import UNet

from task4_instaflow.instaflow_dataset import (
    InstaFlowDataset,
    get_instaflow_data_iterator,
)

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    assert args.use_cfg, "In Assignment 7, we train with CFG setup only."

    # Create save directory
    save_dir = Path(f"results/instaflow-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load InstaFlow distillation dataset
    print(f"Loading distillation dataset from {args.distill_data_path}")
    distill_dataset = InstaFlowDataset(
        args.distill_data_path,
        use_cfg=args.use_cfg
    )

    train_dl = torch.utils.data.DataLoader(
        distill_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    train_it = get_instaflow_data_iterator(train_dl)

    # Get image resolution and num_classes from dataset metadata
    image_resolution = distill_dataset.metadata.get("image_resolution", 64)
    num_classes = distill_dataset.metadata.get("num_classes", None)

    print(f"Image resolution: {image_resolution}")
    if args.use_cfg:
        print(f"Number of classes: {num_classes}")

    # Set up the scheduler (same as base FM)
    fm_scheduler = FMScheduler(sigma_min=args.sigma_min)

    # Initialize student network (same architecture as teacher)
    network = UNet(
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=num_classes,
    )

    fm = FlowMatching(network, fm_scheduler)
    fm = fm.to(config.device)

    # Same optimizer and scheduler as base FM
    optimizer = torch.optim.Adam(fm.network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    print(f"Starting InstaFlow training for {config.train_num_steps} steps...")
    print("Goal: Learn to generate images in ONE STEP from noise to data")

    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                fm.eval()
                # Plot loss curve
                plt.plot(losses)
                plt.xlabel('Training Step')
                plt.ylabel('Distillation Loss')
                plt.title('InstaFlow Training Loss')
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                # Generate sample images with ONE-STEP inference
                shape = (4, 3, fm.image_resolution, fm.image_resolution)
                if args.use_cfg:
                    class_label = torch.tensor([1, 1, 2, 3]).to(config.device)
                    # ONE-STEP GENERATION
                    samples = fm.sample(
                        shape,
                        class_label=class_label,
                        guidance_scale=7.5,
                        num_inference_timesteps=1,  # ONE STEP!
                        verbose=False
                    )
                else:
                    samples = fm.sample(shape, num_inference_timesteps=1, verbose=False)

                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                # Save checkpoint
                fm.save(f"{save_dir}/last.ckpt")
                fm.train()

            # Load batch from distillation dataset
            if args.use_cfg:
                x_0, x_1, label = next(train_it)
                x_0, x_1, label = x_0.to(config.device), x_1.to(config.device), label.to(config.device)
            else:
                x_0, x_1 = next(train_it)
                x_0, x_1 = x_0.to(config.device), x_1.to(config.device)
                label = None

            ######## TODO ########
            # DO NOT change the code outside this part.
            # Implement InstaFlow one-step distillation training:
            #
            # The goal is to train a student model that can generate high-quality images
            # in ONE STEP from noise to data. The student learns from teacher-generated
            # pairs (x_0, x_1) where x_1 is already a high-quality sample.
            #
            # Training objective:
            # The student learns the velocity field that enables one-step generation.
            # We use the same CFM loss but on distillation pairs from the teacher.
            #
            # Loss: E[||v_θ(ψ_t(x_0|x_1); t) - (x_1 - (1 - σ_min)x_0)||²]
            #
            # Key insight: The student learns on teacher's high-quality outputs,
            # and the learned velocity field enables direct one-step mapping.
            #
            # Hint 1: Use fm.get_loss(x_1, x0=x_0, class_label=label)
            # Hint 2: The loss is the same as rectified flow, but the training
            #         enables ONE-STEP inference due to teacher's quality
            # Hint 3: Some implementations add additional losses (perceptual, GAN)
            #         but for this assignment, CFM loss is sufficient

            # Compute distillation loss
            if args.use_cfg:
                loss = fm.get_loss(x_1, class_label=label, x0=x_0)
            else:
                loss = fm.get_loss(x_1, x0=x_0)

            ######################

            pbar.set_description(f"Loss: {loss.item():.4f}")

            # Optimization step
            optimizer.zero_grad()
            loss.backward()

            # Optional: Gradient clipping for stability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(fm.network.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)

    print(f"Training completed! Final checkpoint saved at {save_dir}/last.ckpt")
    print("\nYou can now generate images with ONE STEP:")
    print(f"python image_fm_todo/sampling.py \\")
    print(f"  --use_cfg \\")
    print(f"  --ckpt_path {save_dir}/last.ckpt \\")
    print(f"  --save_dir results/instaflow_samples \\")
    print(f"  --num_inference_steps 1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train InstaFlow one-step generator")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--distill_data_path", type=str, required=True,
                        help="Path to distillation dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="Number of training steps",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--sigma_min", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping (0 to disable)")
    args = parser.parse_args()
    main(args)
