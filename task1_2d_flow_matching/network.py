import math
from typing import List, Optional # 確保匯入 Optional

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


class TimeLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.fc(x)
        alpha = self.time_embedding(t).view(-1, self.dim_out)

        return alpha * x


class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int
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
        
        # 1. 建立時間嵌入 (Time Embedding)
        # 我們將使用 TimeEmbedding 模組。我們需要為它選擇一個維度。
        # 這裡我們使用第一個隱藏層維度 dim_hids[0]。
        # `num_timesteps` 參數在此處不直接使用，但 TimeEmbedding 內部會處理。
        self.time_emb = TimeEmbedding(dim_hids[0])
        
        # 2. 建立一個 nn.Sequential 模型
        # 這是一個標準的 MLP (多層感知器)
        self.model = nn.Sequential()
        
        # 我們的策略是將輸入 x 和時間嵌入 t_emb 串聯(concatenate)起來。
        # 所以第一層的輸入維度是 dim_in + dim_hids[0]
        all_dims = [dim_in + dim_hids[0]] + dim_hids + [dim_out]

        # 3. 迴圈建立 MLP 的所有層
        for i in range(len(all_dims) - 2):
            # 加入線性層
            self.model.add_module(
                f"layer_{i}_linear", nn.Linear(all_dims[i], all_dims[i+1])
            )
            # 加入 SiLU 激活函式
            self.model.add_module(f"layer_{i}_act", nn.SiLU())
            
        # 4. 加入最後的輸出層
        # 輸出層沒有激活函式，因為我們是直接預測速度向量
        self.model.add_module(
            "layer_out_linear", nn.Linear(all_dims[-2], all_dims[-1])
        )
        ######################
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_label: Optional[torch.Tensor] = None):
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
            class_label: (Task 2) conditional label. Ignored in Task 1.
                         We add it here to match the call signature in fm.py.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        
        # 1. 取得時間嵌入
        t_emb = self.time_emb(t) # (batch_size, dim_hids[0])
        
        # 2. 將 x 和時間嵌入 t_emb 串聯
        # x.shape is (batch_size, dim_in)
        # t_emb.shape is (batch_size, dim_hids[0])
        # xt.shape will be (batch_size, dim_in + dim_hids[0])
        xt = torch.cat([x, t_emb], dim=-1)
        
        # 3. 將串聯後的張量傳入模型
        # model(xt) 的輸出 shape 為 (batch_size, dim_out)
        return self.model(xt)
        ######################