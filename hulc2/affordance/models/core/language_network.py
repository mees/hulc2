from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn


class SBert(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.model = SentenceTransformer(weights)

    def forward(self, x: List, show_progress_bar: bool = False) -> torch.Tensor:
        emb = self.model.encode(x, convert_to_tensor=True, show_progress_bar=show_progress_bar)
        return torch.unsqueeze(emb, 1)
