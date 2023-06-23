# Vision networks
from hulc2.affordance.models.language_encoders.bert_lang_encoder import BERTLang

# Language encoders
from hulc2.affordance.models.language_encoders.clip_lang_encoder import CLIPLang
from hulc2.affordance.models.language_encoders.distilbert_lang_encoder import DistilBERTLang
from hulc2.affordance.models.language_encoders.sbert_lang_encoder import SBertLang
from hulc2.affordance.models.visual_lang_encoders.r3m_rn18 import R3M
from hulc2.affordance.models.visual_lang_encoders.rn50_clip_lingunet import CLIPLingUNet
from hulc2.affordance.models.visual_lang_encoders.rn50_unet import RN50LingUNet
from hulc2.affordance.models.visual_lang_encoders.rn_lingunet import RNLingunet

lang_encoders = {"clip": CLIPLang, "bert": BERTLang, "distilbert": DistilBERTLang, "sbert": SBertLang}

vision_encoders = {
    # Lang Nets
    "clip": CLIPLingUNet,
    "rn": RNLingunet,  # RN50LingUNet,
    "rn18": RNLingunet,
    "r3m_rn18": R3M,
}

# Depth estimatiom models
from hulc2.affordance.models.depth.depth_gaussian import DepthEstimationGaussian
from hulc2.affordance.models.depth.depth_logistics import DepthEstimationLogistics

deth_est_nets = {
    # Depth Nets
    "gaussian": DepthEstimationGaussian,
    "logistic": DepthEstimationLogistics,
}
