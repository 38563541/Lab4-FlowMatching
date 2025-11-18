"""
Generate Reflow Dataset for Rectified Flow Training (Task 3)
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


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save configuration
    config = vars(args)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = f"cuda:{args.gpu}"

    # Load pretrained Flow Matching model
    print(f"Loading checkpoint from {args.ckpt_path}")
    # Load with weights_only=False to avoid pickle error
    fm = FlowMatching(None, None)
    fm.load(args.ckpt_path)
    fm.eval()
    fm = fm.to(device)
    
    num_classes = 0
    if args.use_cfg:
        if hasattr(fm.network, 'use_cfg') and fm.network.use_cfg:
            num_classes = fm.network.class_embedding.num_embeddings - 1
            print(f"Using CFG with {num_classes} classes (including null class)")
        else:
            print("Warning: use_cfg is True but model might not support it.")

    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    print(f"Generating {total_num_samples} reflow pairs with {args.num_inference_steps} ODE steps...")

    # =====================================================
    # 關鍵修正：加入 torch.no_grad() 避免記憶體爆炸
    # =====================================================
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating reflow dataset"):
            sidx = i * args.batch_size
            eidx = min(sidx + args.batch_size, total_num_samples)
            B = eidx - sidx

            # 1. 定義 x_0 與 labels
            shape = (B, 3, fm.image_resolution, fm.image_resolution)
            x_0 = torch.randn(shape).to(device)

            if args.use_cfg and num_classes > 0:
                labels = torch.randint(1, num_classes + 1, (B,)).to(device)
            else:
                labels = None

            # 2. 執行手動積分 (產生 z_1)
            zt = x_0.clone()
            timesteps = [j / args.num_inference_steps for j in range(args.num_inference_steps)]

            for t_idx, t_val in enumerate(timesteps):
                t = torch.full((B,), t_val, device=device)
                t_next_val = timesteps[t_idx + 1] if t_idx < len(timesteps) - 1 else 1.0
                
                # 計算 dt
                dt = t_next_val - t_val
                # Expand dt (scalar to tensor with correct shape)
                dt_tensor = torch.full((B, 1, 1, 1), dt, device=device)

                # 預測速度
                if args.use_cfg and labels is not None:
                    v_uncond = fm.network(zt, t, class_label=None)
                    v_cond = fm.network(zt, t, class_label=labels)
                    vt = v_uncond + args.cfg_scale * (v_cond - v_uncond)
                else:
                    vt = fm.network(zt, t)

                # Euler Step: z_{t+1} = z_t + v_t * dt
                zt = zt + vt * dt_tensor

            z_1 = zt

            # 3. 儲存資料
            for j in range(B):
                sample_idx = sidx + j
                torch.save(x_0[j].cpu(), save_dir / f"{sample_idx:06d}_x0.pt")
                torch.save(z_1[j].cpu(), save_dir / f"{sample_idx:06d}_z1.pt")

                if labels is not None:
                    torch.save(labels[j].cpu(), save_dir / f"{sample_idx:06d}_label.pt")

                if args.save_images and sample_idx < 100:
                    img_z1 = tensor_to_pil_image(z_1[j].cpu(), single_image=True)
                    img_z1.save(save_dir / f"{sample_idx:06d}_z1.png")

    print(f"Reflow dataset saved to {save_dir}")
    print(f"Total samples: {total_num_samples}")

    metadata = {
        "num_samples": total_num_samples,
        "num_inference_steps": args.num_inference_steps,
        "use_cfg": args.use_cfg,
        "guidance_scale": args.cfg_scale if args.use_cfg else None,
        "image_resolution": fm.image_resolution,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="data/afhq_reflow")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
