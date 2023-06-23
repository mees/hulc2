import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class LangEncoder(nn.Module):
    def __init__(self, freeze_backbone=True, pretrained=True) -> None:
        super(LangEncoder, self).__init__()
        self.freeze_backbone = freeze_backbone
        self.pretrained = pretrained
        self._load_model()

    def _load_model(self):
        raise NotImplementedError()

    def encode_text(self, x):
        """
        Returns:
            - text_encodings
            - text_embeddings
            - text_mask
        """
        raise NotImplementedError()
