import torch
import torch.nn as nn

from hulc2.affordance.models.core.clip import build_model, load_clip, tokenize
from hulc2.affordance.models.language_encoders.base_lang_encoder import LangEncoder


class CLIPLang(LangEncoder):
    def __init__(self, freeze_backbone=True, pretrained=True) -> None:
        super(CLIPLang, self).__init__(freeze_backbone, pretrained)

    def _load_model(self):
        model, _ = load_clip("RN50", jit=False)
        _clip_rn50 = build_model(model.state_dict())
        del model
        if self.freeze_backbone:
            for param in _clip_rn50.parameters():
                param.requires_grad = False
        #     for param in _clip_rn50.layer4.parameters():
        #         param.requires_grad = True
        else:
            _clip_rn50 = _clip_rn50.float()
        # modules = list(net.children())[:-1]
        self.model = _clip_rn50

    def encode_text(self, x):
        with torch.set_grad_enabled(not self.freeze_backbone):
            tokens = tokenize(x)
            tokens = tokens.to(self.model.positional_embedding.device)
            text_feat, text_emb = self.model.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)
        return text_feat, text_emb, text_mask
