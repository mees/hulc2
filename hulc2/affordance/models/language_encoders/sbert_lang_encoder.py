import numpy as np
import torch
from torch import nn, Tensor

from sentence_transformers import SentenceTransformer
from typing import List
from tqdm.autonotebook import trange
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SBertLang(nn.Module):
    def __init__(self, nlp_model: str, freeze_backbone=True) -> None:
        super(SBertLang, self).__init__()
        self.freeze_backbone = freeze_backbone
        self.model = SentenceTransformer(nlp_model)
        _embd_dim = 384
        # self.text_fc = nn.Linear(_embd_dim, 1024)

    def forward(self, x: List) -> torch.Tensor:
        enc = self.encode(x)
        # enc = self.text_fc(enc)
        return enc # torch.unsqueeze(enc, 1)

    def encode(self, sentences: List[str],
               normalize_embeddings: bool = False) -> Tensor:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           A stacked tensor is returned
        """
        if self.freeze_backbone:
            self.model.eval()

        all_embeddings = []
        length_sorted_idx = np.argsort([-self.model._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        features = self.model.tokenize(sentences_sorted)
        features = self.batch_to_device(features, self.model._target_device)

        with torch.set_grad_enabled(not self.freeze_backbone):
            out_features = self.model.forward(features)
            embeddings = out_features["sentence_embedding"]
            embeddings = embeddings.detach()
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings)

        # undo sort and convert to tensor
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = torch.stack(all_embeddings)
        return all_embeddings

    def batch_to_device(self, batch, target_device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], Tensor):
                batch[key] = batch[key].to(target_device)
        return batch