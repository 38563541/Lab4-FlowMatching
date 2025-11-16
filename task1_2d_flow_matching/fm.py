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
        ts = (
            np.random.choice(np.arange(self.num_train_timesteps), batch_size)
            / self.num_train_timesteps
        )
        return torch.from_numpy(ts)

    def compute_psi_t(self, x1, t, x):
        """
        Compute the conditional flow psi_t(x | x_1).

        Note that time flows in the opposite direction compared to DDPM/DDIM.
        As t moves from 0 to 1, the probability paths shift from a prior distribution p_0(x)
        to a more complex data distribution p_1(x).

        Input:
            x1 (`torch.Tensor`): Data sample from the data distribution (p_1).
            t (`torch.Tensor`): Timestep in [0,1).
            x (`torch.Tensor`): Sample from the prior distribution (p_0).
        Output:
            psi_t (`torch.Tensor`): The interpolated sample x_t.
        """
        t = expand_t(t, x1)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute psi_t(x)
        # 這是 x_t = (1-t) * x0 + t * x1 的路徑
        # 在此函式中, x (來自 p_0) 就是 x0, x1 (來自 p_1) 就是 x1
        x0 = x
        psi_t = (1.0 - t) * x0 + t * x1
        ######################

        return psi_t

    def step(self, xt, vt, dt):
        """
        The simplest ode solver as the first-order Euler method:
        x_next = xt + dt * vt
        """
        print("updated")

        ######## TODO ########
        # DO NOT change the code outside this part.
        # implement each step of the first-order Euler method.
        # x_next = xt + v(xt, t) * dt
        
        # --------------------
        # 這是我們的修正
        # --------------------
        # 我們必須將 dt [shape=(500,)] 擴展為 [shape=(500, 1)]
        # 才能與 xt [shape=(500, 2)] 和 vt [shape=(500, 2)] 
        # 正確地進行廣播 (broadcasting)
        dt = expand_t(dt, xt)
        # --------------------
        
        x_next = xt + vt * dt
        ######################

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
        t = self.fm_scheduler.uniform_sample_t(batch_size).to(x1)
        if x0 is None:
            x0 = torch.randn_like(x1)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Implement the CFM objective.
        
        # 1. 計算 x_t (插值樣本)
        # xt = (1-t) * x0 + t * x1
        xt = self.conditional_psi_sample(x1, t, x0=x0)
        
        # 2. 計算 u_t (目標向量場)
        # u_t = d(xt) / dt = x1 - x0
        ut = x1 - x0
        
        # 3. 取得模型預測的向量場 v_theta(xt, t)
        if class_label is not None:
            model_out = self.network(xt, t, class_label=class_label)
        else:
            # Task 1 將會執行這個路徑
            model_out = self.network(xt, t)
            
        # 4. 計算 L2 Loss: || v_theta - u_t ||^2
        loss = F.mse_loss(model_out, ut)
        ######################

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
        # x_T 在這裡是 x_0 (t=0), 從純雜訊開始
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
        # 將 timesteps 轉換為 tensor
        timesteps = [torch.tensor([t] * x_T.shape[0]).to(x_T) for t in timesteps]
        pbar = tqdm(timesteps) if verbose else timesteps
        
        xt = x_T # xt 是目前的 x, 從 x_0 (x_T) 開始
        for i, t in enumerate(pbar):
            # t_next 是下一個時間點
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else torch.ones_like(t)
            
            # dt 是時間間隔
            dt = t_next - t

            ######## TODO ########
            # Complete the sampling loop
            
            # 1. 預測速度 vt = v_theta(xt, t)
            if do_classifier_free_guidance:
                # 取得有條件的預測
                v_cond = self.network(xt, t, class_label=class_label)
                # 取得無條件的預測 (將 class_label 設為 None)
                v_uncond = self.network(xt, t, class_label=None) # 假設 network 支援 class_label=None
                
                # 執行 CFG 混合
                vt = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                # Task 1 將執行這個路徑
                vt = self.network(xt, t)
            
            # 2. 使用 Euler step 計算 xt_next
            # x_{t+dt} = x_t + v_t * dt
            xt_next = self.fm_scheduler.step(xt, vt, dt)
            
            # 3. 更新 xt
            xt = xt_next
            ######################

            traj[-1] = traj[-1].cpu()
            traj.append(xt.clone().detach())
            
        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "fm_scheduler": self.fm_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.fm_scheduler = hparams["fm_scheduler"]

        self.load_state_dict(state_dict)