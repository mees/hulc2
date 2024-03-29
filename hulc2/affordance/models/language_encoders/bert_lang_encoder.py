import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

from hulc2.affordance.models.language_encoders.base_lang_encoder import LangEncoder


class BERTLang(LangEncoder):
    def __init__(self, freeze_backbone=True, pretrained=True) -> None:
        super(BERTLang, self).__init__(freeze_backbone, pretrained)

    def _load_model(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if self.pretrained:
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        else:
            distilbert_config = BertConfig()
            self.text_encoder = BertModel(distilbert_config)
        _embd_dim = 768
        self.text_fc = nn.Linear(_embd_dim, 1024)

    def encode_text(self, x):
        with torch.set_grad_enabled(not self.freeze_backbone):
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
            input_ids = input_ids.to(self.text_encoder.device)
            attention_mask = attention_mask.to(self.text_encoder.device)
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            text_encodings = text_embeddings.last_hidden_state.mean(1)

        text_feat = self.text_fc(text_encodings)
        text_mask = torch.ones_like(input_ids)  # [1, max_token_len]
        return text_feat, text_embeddings.last_hidden_state, text_mask
