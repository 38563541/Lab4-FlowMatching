import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        # 時間嵌入輸出維度必須與這一層的 output 維度一致，才能進行 element-wise 乘法
        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # 1. 空間特徵轉換
        x = self.fc(x)
        
        # 2. 取得時間特徵並 reshape
        alpha = self.time_embedding(t).view(-1, self.dim_out)

        # 3. 乘法機制 (Scaling/Gating)
        return alpha * x


class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int
    ):
        super().__init__()
        """
        (TODO) Build a noise estimating network.
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hids = dim_hids
        self.num_timesteps = num_timesteps

        self.layers = nn.ModuleList()
        input_dim = dim_in
        
        # 建立隱藏層
        for output_dim in dim_hids:
            self.layers.append(TimeLinear(input_dim, output_dim, num_timesteps))
            input_dim = output_dim

        # 建立輸出層 (注意：這個版本的輸出層也是 TimeLinear)
        self.output_layer = TimeLinear(input_dim, dim_out, num_timesteps)   
        ######################
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_label=None):
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data
            t: the timestep
            class_label: ignored in this simple implementation
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        
        # 通過隱藏層 (TimeLinear -> SiLU)
        for layer in self.layers:
            x = layer(x, t)
            x = F.silu(x)
            
        # 通過輸出層 (TimeLinear, 無 Activation)
        x = self.output_layer(x, t)
        
        ######################
        return x

    @property
    def image_resolution(self):
        return None