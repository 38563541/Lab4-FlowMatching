"""
Train DDPM Teacher Model for InstaFlow Distillation (Task 4)

This script trains a DDPM model that will be used as a teacher for distillation.
The trained DDPM model generates high-quality images through iterative denoising.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.append('.')
from image_common.dataset import AFHQDataModule, get_data_iterator, tensor_to_pil_image
from image_common.ddpm_teacher.model import DiffusionModule
from image_common.ddpm_teacher.scheduler import DDPMScheduler
from image_common.ddpm_teacher.network import UNet

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = vars(args)
    device = f"cuda:{args.gpu}"

    now = get_current_time()
    save_dir = Path(f"results/ddpm_teacher-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(args.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Setup dataset
    image_resolution = 64
    ds_module = AFHQDataModule(
        "./data",
        batch_size=args.batch_size,
        num_workers=4,
        max_num_images_per_cat=args.max_num_images_per_cat,
        image_resolution=image_resolution
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    # Setup DDPM scheduler
    num_train_timesteps = args.num_train_timesteps
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        mode='linear',
    ).to(device)

    # Setup network
    network = UNet(
        T=num_train_timesteps,
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=2,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    # Setup diffusion model
    ddpm = DiffusionModule(network, ddpm_scheduler, predictor=args.predictor)
    ddpm = ddpm.to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / args.warmup_steps, 1.0)
    )

    step = 0
    losses = []

    print(f"Training DDPM teacher model...")
    print(f"Predictor: {args.predictor}")
    print(f"Use CFG: {args.use_cfg}")
    print(f"Total steps: {args.train_num_steps}")

    with tqdm(initial=step, total=args.train_num_steps) as pbar:
        while step < args.train_num_steps:
            # Logging and sampling
            if step % args.log_interval == 0:
                ddpm.eval()

                # Save loss plot
                if len(losses) > 0:
                    plt.plot(losses)
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    plt.title('DDPM Training Loss')
                    plt.savefig(f"{save_dir}/loss.png")
                    plt.close()

                # Generate samples
                shape = (4, 3, ddpm.image_resolution, ddpm.image_resolution)
                if args.use_cfg:
                    class_label = torch.tensor([1, 1, 2, 3]).to(device)
                    samples = ddpm.sample(
                        4,
                        class_label=class_label,
                        guidance_scale=args.guidance_scale
                    )
                else:
                    samples = ddpm.sample(4)

                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                # Save checkpoint
                ddpm.save(f"{save_dir}/last.ckpt")
                print(f"\nStep {step}: Saved checkpoint and samples")

                ddpm.train()

            # Training step
            img, label = next(train_it)
            img, label = img.to(device), label.to(device)

            if args.use_cfg:
                loss = ddpm.get_loss(img, class_label=label)
            else:
                loss = ddpm.get_loss(img)

            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)

    # Final save
    ddpm.save(f"{save_dir}/final.ckpt")
    print(f"\nTraining completed! Final checkpoint saved to {save_dir}/final.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM Teacher Model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=50000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=3000,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="number of diffusion timesteps")
    parser.add_argument("--beta_1", type=float, default=1e-4, help="beta_1 for DDPM scheduler")
    parser.add_argument("--beta_T", type=float, default=0.02, help="beta_T for DDPM scheduler")
    parser.add_argument("--predictor", type=str, default="noise", choices=["noise", "x0", "mean"],
                        help="type of predictor (noise, x0, or mean)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cfg", action="store_true", help="use classifier-free guidance")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale for sampling")

    args = parser.parse_args()
    main(args)
