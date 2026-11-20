import json
from pathlib import Path
import draccus
import torch
from rho.common.types import PolicyFeature,FeatureType
from rho.policies.rho.configuration_rho import RhoConfig
from rho.policies.rho.rho_policy import RhoPolicy
from rho.checkpoints import load_bundle_state_dict
ROOT=Path(__file__).resolve().parents[1]
def build():
    d=json.loads((ROOT/'work/sh5/base_policy.json').read_text());d.pop('type')
    d.update(sh5_tactile=True,max_state_dim=144,max_action_dim=54,train_expert_only=True,
             freeze_vlm_backbone=True,freeze_vision_encoder=True,attention_implementation='sdpa',dropout_p=.02,
             log_hidden_state_stats=False)
    cfg=draccus.decode(RhoConfig,d)
    cfg.feature_dict={'observation.state':PolicyFeature(FeatureType.STATE,(144,)),
                      'action':PolicyFeature(FeatureType.ACTION,(54,)),
                      'observation.image.0':PolicyFeature(FeatureType.VISUAL,(3,188,336))}
    policy=RhoPolicy(cfg)
    weights=load_bundle_state_dict(ROOT/'pretrained/rho-base')
    weights=policy._remap_checkpoint_keys(weights)
    prefixes=('model.flow_model.state_projector.','model.flow_model.action_in_proj.','model.flow_model.action_head.')
    # New embodiment has different joint semantics: reinitialize only these adapters.
    fresh=policy.state_dict();replaced=[]
    for k in list(weights):
        if k.startswith(prefixes):del weights[k];replaced.append(k)
    weights.update({k:v for k,v in fresh.items() if k.startswith(prefixes)})
    policy.load_state_dict(weights,strict=True)
    del weights,fresh
    policy.to('cuda')
    # FP32 trainable weights/moments avoid BF16 rounding erasing small AdamW updates.
    for parameter in policy.parameters():
        if parameter.requires_grad: parameter.data = parameter.data.float()
    policy.model.flow_model.vlm_backbone.eval()
    return policy,replaced
