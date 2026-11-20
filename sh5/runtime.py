"""Offline inference from this run's trusted full checkpoint; returns absolute radians."""
import cv2
import numpy as np
import torch
from rho.policies.rho.rho_policy import RhoPolicy
class Runtime:
    def __init__(self,path):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        self.stats=checkpoint['stats'];self.prompts=checkpoint['prompts'];self.policy=RhoPolicy(checkpoint['config'])
        for parameter in self.policy.parameters():
            if parameter.requires_grad:parameter.data=parameter.data.float()
        self.policy.load_state_dict(checkpoint['model'],strict=True);self.policy.cuda().eval()
    @torch.inference_mode()
    def predict(self,joints,tactile_left,tactile_right,head_rgb,task_id):
        joints=np.asarray(joints,np.float32);touch=np.concatenate([tactile_left,tactile_right]).astype(np.float32)
        if joints.shape!=(54,) or touch.shape!=(90,) or head_rgb.shape!=(720,1280,3) or head_rgb.dtype!=np.uint8:
            raise ValueError('Expected joints54, left/right45 and RGB uint8 720x1280')
        if not np.isfinite(joints).all() or not np.isfinite(touch).all():raise ValueError('nonfinite inputs')
        if task_id not in (0,1):raise ValueError('task_id must be 0 (905/right) or 1 (943/left)')
        prompt=self.prompts[task_id]
        s=self.stats;state=np.concatenate([(joints-s['state_mean'])/s['state_std'],(touch-s['tactile_mean'])/s['tactile_std']])
        image=cv2.resize(head_rgb,(336,188),interpolation=cv2.INTER_LINEAR)
        batch={'observation.state':torch.from_numpy(state)[None].cuda(),'observation.image.0':torch.from_numpy(image).permute(2,0,1)[None].cuda(),'task':[prompt]}
        with torch.autocast('cuda',dtype=torch.bfloat16):actions=self.policy.sample_actions(batch)['actions'][0].float().cpu().numpy()
        actions=actions*s['action_std']+s['action_mean']
        if actions.shape!=(50,54) or not np.isfinite(actions).all():raise ValueError('invalid predicted action chunk')
        return actions
