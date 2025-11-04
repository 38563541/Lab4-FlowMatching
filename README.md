<div align="center">
  <h1>
  Flow Matching
  </h1>
  <p>
    <b>NYCU: Image and Video Generation (2025 Fall)</b><br>
    Programming Assignment 4
  </p>
</div>

<div align="center">
  <p>
    Instructor: <b>Yu-Lun Liu</b><br>
    TA: <b>Jie-Ying Lee</b>, <b>Ying-Huan Chen</b>
  </p>
</div>

<div align=center>
   <img src="./assets/trajectory_visualization.png">
</div>

---

## Overview

This assignment consists of **4 tasks** with a total score of **100 points**:

- **Tasks 1-4 Implementation**: 85 points
- **Report**: 15 points

### Grading

| Task | Points | Description |
|------|--------|-------------|
| Task 1 | 20 pts | 2D Flow Matching Implementation |
| Task 2 | 20 pts | Image Flow Matching with CFG |
| Task 3 | 20 pts | Rectified Flow |
| Task 4 | 25 pts | InstaFlow One-Step Generation |
| **Report** | **15 pts** | **Analysis, results, and discussion** |
| **Total** | **100 pts** | |

---

## Project Structure

The project is organized into four main tasks, each contained within its own directory:

- `task1_2d_flow_matching/`: Task 1 - 2D visualization of Flow Matching.
- `task2_image_flow_matching/`: Task 2 - Image generation with Flow Matching.
- `task3_rectified_flow/`: Task 3 - Rectified Flow for faster sampling.
- `task4_instaflow/`: Task 4 - InstaFlow for one-step generation.
- `image_common/`: Shared code for the image-based tasks (Tasks 2, 3, and 4).
- `fid/`: Tools for Frechet Inception Distance (FID) evaluation.

## Setup

Install the required packages:
```bash
pip install -r requirements.txt
```

**Note:** This assignment depends on implementations from previous assignments. You may need to copy the checkpoint files of your trained models from Assignment 1 (DDPM) as teachers for Task 4 (InstaFlow).

## Recommended Reading

To better understand the concepts and implementations in this assignment, we encourage you to read the following papers:

- **Task 1 & 2: Flow Matching**
  - Lipman, Y., et al. (2022). "Flow Matching for Generative Modeling." [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

- **Task 3: Rectified Flow**
  - Liu, Q., et al. (2022). "Rectified Flow: A Marginal Preserving Approach to Optimal Transport." [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

- **Task 4: InstaFlow**
  - Liu, X., et al. (2023). "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation." [arXiv:2309.06380](https://arxiv.org/abs/2309.06380)

- **DDPM (Teacher Model)**
  - Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

## Tasks

### Task 1: 2D Flow Matching (20 points)

- **Directory**: `task1_2d_flow_matching/`
- **Description**: Implement and visualize Flow Matching on 2D toy datasets.

#### 📝 TODO

You need to implement the following in `task1_2d_flow_matching/`:

1. **`fm.py`**:
   - ✏️ `FMScheduler.compute_psi_t()`: Implement conditional flow ψ_t(x|x_1)
   - ✏️ `FMScheduler.step()`: Implement Euler ODE solver
   - ✏️ `FlowMatching.get_loss()`: Implement CFM training loss
   - ✏️ `FlowMatching.sample()`: Implement sampling with optional CFG

2. **`network.py`**:
   - ✏️ Implement `SimpleNet` (from Assignment 1)

#### 🚀 How to Run

Open and execute the Jupyter notebook:
```bash
jupyter notebook task1_2d_flow_matching/fm_tutorial.ipynb
```

The notebook will train the model and visualize trajectories on toy datasets (Swiss-roll).

#### 📊 What to Report

- Visualizations of learned trajectories
- Chamfer Distance metric
- Comparison with different numbers of inference steps

### Task 2: Image Generation with Flow Matching (20 points)

- **Directory**: `task2_image_flow_matching/`
- **Description**: Implement Flow Matching for image generation on the AFHQ dataset with Classifier-Free Guidance.

#### 📝 TODO

You need to implement the following in `image_common/`:

1. **`fm.py`** (Same TODOs as Task 1, but for images):
   - ✏️ `FMScheduler.compute_psi_t()`: Implement conditional flow ψ_t(x|x_1)
   - ✏️ `FMScheduler.step()`: Implement Euler ODE solver
   - ✏️ `FlowMatching.get_loss()`: Implement CFM training loss
   - ✏️ `FlowMatching.sample()`: Implement sampling with CFG support

2. **`network.py`** (Classifier-Free Guidance):
   - ✅ **COMPLETED**: Random null conditioning during training
   - ✅ **COMPLETED**: Class embedding integration

#### 🚀 How to Run

**Step 1: Download Dataset**
```bash
python -m image_common.dataset  # Downloads AFHQ dataset
```

**Step 2: Train Model**
```bash
python -m task2_image_flow_matching.train --use_cfg
```

Training parameters (fixed):
- Batch size: 16
- Training steps: 100,000
- Learning rate: 2e-4 with warmup
- CFG dropout: 0.1

**Step 3: Generate Samples**
```bash
python -m image_common.sampling \
  --use_cfg \
  --ckpt_path ${CKPT_PATH} \
  --save_dir ${SAVE_DIR_PATH} \
  --num_inference_steps 20
```

**Step 4: Evaluate FID**
```bash
python -m fid.measure_fid data/afhq/eval ${SAVE_DIR_PATH}
```

#### 📊 What to Report

- Training loss curves
- Generated image samples (with/without CFG)
- FID scores with different CFG scales
- Comparison of inference steps (10, 20, 50)

### Task 3: Rectified Flow (20 points)

- **Directory**: `task3_rectified_flow/`
- **Description**: Implement Rectified Flow to straighten generation trajectories and enable faster sampling through the reflow procedure.

#### 📝 TODO

**Note**: This task has **NO additional TODOs** - you will reuse your implementations from Tasks 1 and 2. The focus is on understanding and applying the reflow procedure.

#### 🚀 How to Run

**Step 1: Generate Reflow Dataset**

Use your trained Task 2 model to generate synthetic training pairs:

```bash
python -m task3_rectified_flow.generate_reflow_data \
  --ckpt_path ${TASK2_CKPT_PATH} \
  --num_samples 50000 \
  --save_dir data/afhq_reflow \
  --use_cfg \
  --num_inference_steps 20
```

This creates (x_0, x_1) pairs where x_1 is generated by your Task 2 model following the ODE.

**Step 2: Train Rectified Flow Model**

Train a new Flow Matching model on the synthetic pairs (same architecture and hyperparameters as Task 2):

```bash
python -m task3_rectified_flow.train_rectified \
  --reflow_data_path data/afhq_reflow \
  --use_cfg \
  --reflow_iteration 1
```

Training parameters (same as Task 2):
- Batch size: 16
- Training steps: 100,000
- Learning rate: 2e-4 with warmup

**Step 3: Generate Samples with Fewer Steps**

Test the rectified flow model with reduced inference steps:

```bash
# Generate with 5 steps
python -m image_common.sampling \
  --use_cfg \
  --ckpt_path ${RECTIFIED_CKPT_PATH} \
  --save_dir results/rectified_5steps \
  --num_inference_steps 5

# Generate with 10 steps
python -m image_common.sampling \
  --use_cfg \
  --ckpt_path ${RECTIFIED_CKPT_PATH} \
  --save_dir results/rectified_10steps \
  --num_inference_steps 10
```

**Step 4: Evaluate FID**

```bash
python -m fid.measure_fid data/afhq/eval results/rectified_5steps
python -m fid.measure_fid data/afhq/eval results/rectified_10steps
```

#### 📊 What to Report

- FID scores with different numbers of inference steps (5, 10, 20)
- Comparison with Task 2 base Flow Matching (same steps)
- Speedup analysis (generation time comparison)
- Discussion: Why does rectified flow enable fewer sampling steps?
- Optional: Visualize trajectory straightness on 2D toy datasets

### Task 4: InstaFlow One-Step Generation (25 points)

- **Directory**: `task4_instaflow/`
- **Description**: Implement InstaFlow for one-step image generation through a two-phase distillation pipeline:
  1. **Phase 1**: Train 2-Rectified Flow from DDPM teacher (straighten generation paths)
  2. **Phase 2**: Distill 2-RF to one-step InstaFlow generator

**Pipeline Overview**:
```
DDPM (α₁=7.5) → 2-Rectified Flow → InstaFlow (α₂=1.5) → ONE-STEP Generation
   Phase 1: Reflow            Phase 2: Distillation
```

**Key Concepts**:
- Two-phase approach prevents blurry images (direct DDPM distillation typically fails)
- Uses different CFG scales: α₁=7.5 (Phase 1), α₂=1.5 (Phase 2)
- Optional LPIPS perceptual loss for improved visual quality
- CFG effect is "baked into" model weights during training

#### 📝 TODO

You need to implement the following in `image_common/instaflow.py`:

1. **`InstaFlowModel.get_loss()`**: Implement one-step distillation objective (Eq. 6 in InstaFlow paper)
   - ✏️ Create t=0 tensor for the entire batch
   - ✏️ Get predicted velocity v(x_0, 0 | class_label) from student network
   - ✏️ Calculate predicted image: x1_pred = x_0 + v_pred
   - ✏️ Compute L2 loss between x1_pred and target x1
   - ✏️ Add optional LPIPS perceptual loss if enabled

2. **`InstaFlowModel.sample()`**: Implement one-step sampling
   - ✏️ Set x_0 = x_T (initial noise)
   - ✏️ Create t=0 tensor
   - ✏️ Get predicted velocity from network (no CFG needed - it's baked in!)
   - ✏️ Calculate final image: x_1 = x_0 + v_pred

**Note**: The two-phase training scripts and 2-RF model are already implemented. Your focus is on completing the InstaFlow model's core functions.

#### 🚀 How to Run

**Prerequisites**:
- DDPM checkpoint from Assignment 1
- Install LPIPS (optional): `pip install lpips`

---

**Phase 1: Train 2-Rectified Flow Teacher**

Create a straight-path teacher model from DDPM:

**Step 1a: Generate 2-RF Training Data**
```bash
python -m task4_instaflow.phase1_generate_2rf_data \
  --ddpm_ckpt_path /path/to/your/ddpm_model.ckpt \
  --num_samples 50000 \
  --save_dir data/afhq_2rf \
  --use_cfg \
  --cfg_scale 7.5
```

This generates (x_0, x_1) pairs where x_1 comes from DDPM with high CFG guidance (α₁=7.5).

**Step 1b: Train 2-Rectified Flow**
```bash
python -m task4_instaflow.phase1_train_2rf \
  --reflow_data_path data/afhq_2rf \
  --use_cfg \
  --train_num_steps 100000
```

Training parameters (same as Task 2):
- Batch size: 16
- Training steps: 100,000
- Learning rate: 2e-4 with warmup

---

**Phase 2: Train InstaFlow One-Step Generator**

Distill 2-RF teacher to one-step student:

**Step 2a: Generate InstaFlow Distillation Data**
```bash
python -m task4_instaflow.generate_instaflow_data \
  --rf2_ckpt_path results/2rf_from_ddpm-XXX/last.ckpt \
  --num_samples 50000 \
  --save_dir data/afhq_instaflow \
  --use_cfg \
  --cfg_scale 1.5
```

This generates (x_0, x_1) pairs where x_1 comes from 2-RF with lower CFG guidance (α₂=1.5).

**Step 2b: Train InstaFlow Student**
```bash
python -m task4_instaflow.train_instaflow \
  --distill_data_path data/afhq_instaflow \
  --use_cfg \
  --use_lpips \
  --train_num_steps 100000
```

Training parameters:
- Batch size: 16
- Training steps: 100,000
- Learning rate: 2e-4 with warmup
- Optional: Add `--use_lpips` for perceptual loss

---

**Evaluation and Sampling**

**Step 3: Evaluate One-Step Generation**
```bash
python -m task4_instaflow.evaluate_instaflow \
  --rf2_ckpt_path results/2rf_from_ddpm-XXX/last.ckpt \
  --instaflow_ckpt_path results/instaflow-XXX/last.ckpt \
  --save_dir results/instaflow_eval
```

This compares InstaFlow (1 step) against the 2-RF teacher (20 steps).

**Step 4: Sample with ONE STEP**
```bash
python -m image_common.sampling \
  --use_cfg \
  --ckpt_path results/instaflow-XXX/last.ckpt \
  --save_dir results/instaflow_samples \
  --num_inference_steps 1
```

**Step 5: Measure FID**
```bash
python -m fid.measure_fid data/afhq/eval results/instaflow_samples
```

#### 📊 What to Report

- FID scores for all models:
  - 2-Rectified Flow with 20 steps
  - InstaFlow with 1 step
- Generation speed comparison (samples/second)
- Speedup analysis vs DDPM (1000 steps) and 2-RF (20 steps)
- Visual quality comparison (show sample images)
- Discussion:
  - Why is two-phase training necessary?
  - Effect of LPIPS loss on visual quality
  - Quality-speed tradeoff analysis
  - Why use different CFG scales (α₁=7.5 vs α₂=1.5)?

#### 📝 Notes

- **Training Time**: ~14 hours per phase (100K steps each), total ~28 hours
- **CFG Strategy**:
  - Phase 1 uses α₁=7.5 (high quality from DDPM)
  - Phase 2 uses α₂=1.5 (prevents over-saturation)
  - Inference requires no CFG (baked into weights!)
- **LPIPS Loss**: Optional but may improve visual quality
- **See Also**: `TASK4_IMPLEMENTATION_SUMMARY.md` for detailed technical documentation

---

## Quick Reference

### All Commands Use `python -m`

All scripts use the module execution format for consistent import resolution:

```bash
# Good: Using python -m
python -m task2_image_flow_matching.train --use_cfg
python -m task4_instaflow.phase1_train_2rf --reflow_data_path data/afhq_2rf

# Bad: Direct script execution (may cause import errors)
python task2_image_flow_matching/train.py --use_cfg
```

### File Organization

```
Diffusion-Assignment7-Flow/
├── task1_2d_flow_matching/      # 2D visualization
├── task2_image_flow_matching/    # Base Flow Matching
├── task3_rectified_flow/         # Rectified Flow (FM → 1-RF)
├── task4_instaflow/              # InstaFlow (DDPM → 2-RF → InstaFlow)
│   ├── phase1_generate_2rf_data.py
│   ├── phase1_train_2rf.py
│   ├── generate_instaflow_data.py
│   ├── train_instaflow.py
│   └── evaluate_instaflow.py
├── image_common/                 # Shared components
│   ├── fm.py                    # Flow Matching
│   ├── instaflow.py             # InstaFlow Model
│   ├── network.py               # U-Net with CFG
│   ├── ddpm_teacher/            # DDPM interface
│   └── sampling.py              # Unified sampling
└── fid/                         # FID evaluation
```

---

## Additional Resources

- **Detailed Documentation**: See `TASK4_IMPLEMENTATION_SUMMARY.md`
- **Project Instructions**: See `CLAUDE.md` for comprehensive guidance
- **Issue Tracking**: Check GitHub issues for known problems
- **Papers**: See "Recommended Reading" section above

---

## License

See `LICENSE` file for details.