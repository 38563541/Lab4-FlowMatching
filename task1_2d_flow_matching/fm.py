from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def expand_t(t, x):
    for _ in range(x.ndim - 1):
        t = t.unsqueeze(-1)
    return t


class FMScheduler(nn.Module):
    def __init__(self, num_train_timesteps=1000, sigma_min=0.001):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min

    def uniform_sample_t(self, batch_size) -> torch.LongTensor:
        # ----------------------------------------------------
        # 這是課程提供的程式碼 - 我們假設它正確
        # 它會產生 t = [0, 0.001, 0.002, ..., 0.999]
        # ----------------------------------------------------
        ts = (
            np.random.choice(np.arange(self.num_train_timesteps), batch_size)
            / self.num_train_timesteps
        )
        return torch.from_numpy(ts)

    def compute_psi_t(self, x1, t, x):
        """
        Compute the conditional flow psi_t(x | x_1).
        """
        t = expand_t(t, x1)

        ######## TODO 1 (修正) ########
        # 我們不使用 t，而是使用 sigma_t
        # sigma_t = (1 - sigma_min) * t + sigma_min
        # 這能確保我們的路徑最小從 sigma_min (0.01) 開始，避免 t=0
        
        sigma_t = (1.0 - self.sigma_min) * t + self.sigma_min
        
        # x_t = (1 - sigma_t) * x0 + sigma_t * x1
        x0 = x
        psi_t = (1.0 - sigma_t) * x0 + sigma_t * x1
        ##############################

        return psi_t

    def step(self, xt, vt, dt):
        """
        The simplest ode solver as the first-order Euler method:
        x_next = xt + dt * vt
        """

        ######## TODO 2 (已修正) ########
        # 保持我們先前的 bug fix
        dt = expand_t(dt, xt)
        x_next = xt + vt * dt
        ##############################

        return x_next


class FlowMatching(nn.Module):
    def __init__(self, network: nn.Module, fm_scheduler: FMScheduler, **kwargs):
        super().__init__()
        self.network = network
        self.fm_scheduler = fm_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    def get_loss(self, x1, class_label=None, x0=None):
        """
        The conditional flow matching objective, corresponding Eq. 23 in the FM paper.
        """
        batch_size = x1.shape[0]
        # t 可能是 0
        t = self.fm_scheduler.uniform_sample_t(batch_size).to(x1)
        if x0 is None:
            x0 = torch.randn_like(x1)

        ######## TODO 3 (修正) ########
        # 1. 計算 x_t (使用新的 sigma_t 路徑)
        #    這會呼叫我們修正過的 compute_psi_t
        xt = self.conditional_psi_sample(x1, t, x0=x0)
        
        # 2. 計算 u_t (目標向量場)
        #    如果 x_t = (1 - sigma_t)x0 + sigma_t*x1
        #    其中 sigma_t = (1 - sigma_min) * t + sigma_min
        #    那麼 d(x_t) / dt = d(sigma_t) / dt * (x1 - x0)
        #    d(sigma_t) / dt = (1 - sigma_min)
        #    所以 u_t = (1 - self.fm_scheduler.sigma_min) * (x1 - x0)
        
        sigma_derivative = (1.0 - self.fm_scheduler.sigma_min)
        ut = sigma_derivative * (x1 - x0)
        
        # 3. 取得模型預測的向量場 v_theta(xt, t)
        if class_label is not None:
            model_out = self.network(xt, t, class_label=class_label)
        else:
            model_out = self.network(xt, t)
            
        # 4. 計算 L2 Loss: || v_theta - u_t ||^2
        loss = F.mse_loss(model_out, ut)
        ##############################

        return loss

    def conditional_psi_sample(self, x1, t, x0=None):
        if x0 is None:
            x0 = torch.randn_like(x1)
        return self.fm_scheduler.compute_psi_t(x1, t, x0)

    @torch.no_grad()
    def sample(
        self,
        shape,
        num_inference_timesteps=50,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
        verbose=False,
    ):
        batch_size = shape[0]
        x_T = torch.randn(shape).to(self.device)
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            assert class_label is not None
            assert (
                len(class_label) == batch_size
            ), f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"

        traj = [x_T]

        timesteps = [
            i / num_inference_timesteps for i in range(num_inference_timesteps)
        ]
        timesteps = [torch.tensor([t] * x_T.shape[0]).to(x_T) for t in timesteps]
        pbar = tqdm(timesteps) if verbose else timesteps
        
        xt = x_T
        for i, t in enumerate(pbar):
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else torch.ones_like(t)
            dt = t_next - t

            ######## TODO 4 (已修正) ########
            # 保持我們先前的 bug fix
            if do_classifier_free_guidance:
                v_cond = self.network(xt, t, class_label=class_label)
                v_uncond = self.network(xt, t, class_label=None) 
                vt = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                vt = self.network(xt, t)
            
            xt_next = self.fm_scheduler.step(xt, vt, dt)
            xt = xt_next
            ##############################

            traj[-1] = traj[-1].cpu()
            traj.append(xt.clone().detach())
            
        if return_traj:
            return traj
        else:
            return traj[-1]

    # ... (save 和 load 函式不變) ...