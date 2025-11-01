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

### Task 1: 2D Flow Matching

- **Directory**: `task1_2d_flow_matching/`
- **Description**: Visualize Flow Matching on 2D toy datasets.
- **To run**: Open and execute the Jupyter notebook:
  ```bash
  jupyter notebook task1_2d_flow_matching/fm_tutorial.ipynb
  ```

### Task 2: Image Generation with Flow Matching

- **Directory**: `task2_image_flow_matching/`
- **Description**: Generate images using Flow Matching on the AFHQ dataset.
- **Train**:
  ```bash
  python -m task2_image_flow_matching.train --use_cfg
  ```
- **Sample**:
  ```bash
  python task2_image_flow_matching/sampling.py --use_cfg --ckpt_path ${CKPT_PATH} --save_dir ${SAVE_DIR_PATH}
  ```
- **Evaluate FID**:
  ```bash
  python task2_image_flow_matching/dataset.py  # Prepare dataset
  python fid/measure_fid.py data/afhq/eval ${SAVE_DIR_PATH}
  ```

### Task 3: Rectified Flow

- **Directory**: `task3_rectified_flow/`
- **Description**: Implement Rectified Flow to straighten trajectories and accelerate sampling.
- **Step 1: Generate Reflow Dataset**:
  ```bash
  python -m task3_rectified_flow.generate_reflow_data \
    --ckpt_path ${TASK2_CKPT_PATH} \
    --num_samples 50000 \
    --save_dir data/afhq_reflow \
    --use_cfg \
    --num_inference_steps 20
  ```
- **Step 2: Train Rectified Flow Model**:
  ```bash
  python -m task3_rectified_flow.train_rectified \
    --reflow_data_path data/afhq_reflow \
    --use_cfg \
    --reflow_iteration 1
  ```
- **Step 3: Evaluate with Fewer Steps**:
  ```bash
  python -m image_common.sampling \
    --use_cfg \
    --ckpt_path ${RECTIFIED_CKPT_PATH} \
    --save_dir results/rectified_5steps \
    --num_inference_steps 5
  ```

### Task 4: InstaFlow

- **Directory**: `task4_instaflow/`
- **Description**: Implement InstaFlow for one-step image generation.

#### Using a Flow Matching (FM) Teacher (Default)

- **Step 1: Generate Distillation Dataset**:
  ```bash
  python -m task4_instaflow.generate_instaflow_data \
    --teacher_type fm \
    --ckpt_path ${TEACHER_CKPT_PATH} \
    --num_samples 50000 \
    --save_dir data/afhq_instaflow \
    --use_cfg \
    --num_inference_steps 20
  ```

#### Using a DDPM Teacher (from Lab1-DDPM)

- **Step 1: Generate Distillation Dataset**:
  ```bash
  python -m task4_instaflow.generate_instaflow_data \
    --teacher_type ddpm \
    --ddpm_ckpt_path /path/to/your/ddpm_model.ckpt \
    --num_samples 50000 \
    --save_dir data/afhq_instaflow_ddpm_teacher \
    --use_cfg
  ```

- **Step 2: Train InstaFlow Student**:
  ```bash
  python -m task4_instaflow.train_instaflow \
    --distill_data_path data/afhq_instaflow_ddpm_teacher \
    --use_cfg \
    --train_num_steps 100000
  ```

- **Step 3: Evaluate One-Step Generation**:
  ```bash
  python -m task4_instaflow.evaluate_instaflow \
    --base_ckpt_path ${BASE_FM_CKPT} \
    --instaflow_ckpt_path ${INSTAFLOW_CKPT} \
    --save_dir results/instaflow_eval
  ```

- **Step 4: Sample with ONE STEP**:
  ```bash
  python -m image_common.sampling \
    --use_cfg \
    --ckpt_path ${INSTAFLOW_CKPT} \
    --save_dir results/instaflow_samples \
    --num_inference_steps 1
  ```