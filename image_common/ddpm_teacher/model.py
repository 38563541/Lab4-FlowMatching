from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .scheduler import extract

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler
        self.predictor = kwargs.get("predictor", "noise")


    def get_loss_noise(self, x0, class_label=None, noise=None):
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)  # (B,)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
        eps_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)
        return F.mse_loss(eps_pred, eps)
    
    def get_loss_x0(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # Here we implement the "predict x0" version.
        # 1. Sample a timestep and add noise to get (x_t, noise).
        # 2. Pass (x_t, timestep) into self.network, where the output should represent the clean sample x0_pred.
        # 3. Compute the loss as MSE(predicted x0_pred, ground-truth x0).
        ######################
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)  # (B,)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
        x0_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)
        loss = F.mse_loss(x0_pred, x0)
        return loss
    
    def get_loss_mean(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # Here we implement the "predict mean" version.
        # 1. Sample a timestep and add noise to get (x_t, noise).
        # 2. Pass (x_t, timestep) into self.network, where the output should represent the posterior mean μθ(x_t, t).
        # 3. Compute the *true* posterior mean from the closed-form DDPM formula (using x0, x_t, noise, and scheduler terms).
        # 4. Compute the loss as MSE(predicted mean, true mean).
        ######################
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)  # (B,)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)

        # Predicted mean from network
        mean_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)

        # True posterior mean: μ_q(x_t, x_0) = (sqrt(ᾱ_{t-1}) * β_t / (1 - ᾱ_t)) * x_0 + (sqrt(α_t) * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)) * x_t
        alpha_t = extract(self.var_scheduler.alphas, t, x_t)
        alpha_bar_t = extract(self.var_scheduler.alphas_cumprod, t, x_t)
        beta_t = extract(self.var_scheduler.betas, t, x_t)

        # alpha_bar_{t-1}
        alpha_bar_t_prev = extract(
            torch.cat([torch.tensor([1.0]).to(x_t.device), self.var_scheduler.alphas_cumprod[:-1]]),
            t,
            x_t
        )

        # Compute true posterior mean
        coef1 = (alpha_bar_t_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
        coef2 = (alpha_t.sqrt() * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t)
        mean_true = coef1 * x0 + coef2 * x_t

        loss = F.mse_loss(mean_pred, mean_true)
        return loss
    
    def get_loss(self, x0, class_label=None, noise=None):
        if self.predictor == "noise":
            return self.get_loss_noise(x0, class_label, noise)
        elif self.predictor == "x0":
            return self.get_loss_x0(x0, class_label, noise)
        elif self.predictor == "mean":
            return self.get_loss_mean(x0, class_label, noise)
        else:
            raise ValueError(f"Unknown predictor: {self.predictor}")
    
    
    
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:

            ######## TODO ########
            # Assignment 2. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"

            # Create tensor with null class (0) and real class labels
            null_label = torch.zeros_like(class_label)
            class_label = torch.cat([null_label, class_label], dim=0)
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.
                # Duplicate x_t for both conditional and unconditional predictions
                x_t_input = torch.cat([x_t, x_t], dim=0)
                net_out = self.network(x_t_input, timestep=t.to(self.device), class_label=class_label)

                # Split the output into unconditional and conditional parts
                net_out_uncond, net_out_cond = net_out.chunk(2, dim=0)

                # Apply classifier-free guidance formula: out = out_uncond + guidance_scale * (out_cond - out_uncond)
                net_out = net_out_uncond + guidance_scale * (net_out_cond - net_out_uncond)
                #######################
            else:
                # 如果是 conditional 就傳 class_label，否則就兩個參數
                if class_label is not None:
                    net_out = self.network(x_t, timestep=t.to(self.device), class_label=class_label)
                else:
                    net_out = self.network(x_t, timestep=t.to(self.device))

            x_t_prev = self.var_scheduler.step(x_t, t, net_out, predictor=self.predictor)


            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]


    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
