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
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
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


class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int = 1000
    ):
        super().__init__()
        """
        (TODO) Build a noise estimating network.

        Args:
            dim_in: dimension of input
            dim_out: dimension of output
            dim_hids: dimensions of hidden features
            num_timesteps: number of timesteps
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Build MLP with time conditioning
        
        # 1. 定義時間嵌入層 (Time Embedding)
        # 我們將時間嵌入維度設為第一層隱藏層的大小，方便拼接或相加
        self.time_emb_dim = dim_hids[0]
        self.time_embedding = TimeEmbedding(self.time_emb_dim)

        # 2. 建構 MLP 層
        layers = []
        
        # 輸入層：我們採取 "拼接 (Concatenation)" 策略
        # 輸入維度 = 原始資料維度 (dim_in) + 時間嵌入維度 (time_emb_dim)
        current_dim = dim_in + self.time_emb_dim
        
        # 建立隱藏層
        for hid_dim in dim_hids:
            layers.append(nn.Linear(current_dim, hid_dim))
            layers.append(nn.SiLU()) # SiLU 在 Flow Matching/Diffusion 中表現通常比 ReLU 好
            current_dim = hid_dim # 下一層的輸入是這一層的輸出
            
        self.mlp = nn.Sequential(*layers)

        # 3. 輸出層 (映射回 dim_out，通常是速度向量的維度)
        self.head = nn.Linear(current_dim, dim_out)
        
        ######################

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_label=None):
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        
        # 1. 取得時間嵌入向量
        t_emb = self.time_embedding(t) # Shape: (Batch, time_emb_dim)
        
        # 2. 將時間嵌入與輸入資料拼接 (Concatenate)
        # x shape: (Batch, dim_in)
        # t_emb shape: (Batch, time_emb_dim)
        # result shape: (Batch, dim_in + time_emb_dim)
        x_input = torch.cat([x, t_emb], dim=-1)
        
        # 3. 通過 MLP
        h = self.mlp(x_input)
        
        # 4. 輸出預測結果
        out = self.head(h)
        
        ######################
        
        return out

    @property
    def image_resolution(self):
        # Task 1 是 2D 資料，沒有解析度，回傳 None 即可
        return None