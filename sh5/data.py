"""Read-only SH5 v2.1 adapter. Raw absolute targets; no clamping or cross-episode windows."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
WORKSPACE=Path(__file__).resolve().parents[2]
ROOT=WORKSPACE/'Isaac-GR00T-ttg/datasets/task905_943'
CACHE=WORKSPACE/'hand_intern/workspaces/task905_943_vitacformer/run/cache'
JOINTS=[f'arm_{s}_joint{i}' for s in ('l','r') for i in range(1,8)]+[f'finger_{s}_joint{i}' for s in ('l','r') for i in range(1,21)]
class SH5Dataset(Dataset):
    def __init__(self,split,stats=None,horizon=50):
        self.horizon=horizon; self.split=split; root=ROOT/split
        self.info=json.loads((root/'meta/info.json').read_text())
        assert self.info['fps']==30 and self.info['robot_type']=='ffw_sh5_rev1'
        for k in ['observation.state','action']: assert self.info['features'][k]['names']==JOINTS
        self.source_prompts={d['task_index']:d['task'] for d in map(json.loads,(root/'meta/tasks.jsonl').read_text().splitlines())}
        self.prompts={}
        for task_id,hand in ((0,'right'),(1,'left')):
            expected=f'Use the {hand} hand to pick up the object at the magenta dot from the box and hold it.'
            assert self.source_prompts[task_id]==expected
            self.prompts[task_id]=expected.replace('the object at the magenta dot','a white sanitary pad')
        state=[];action=[];touch=[];anchors=[];tasks=[];offset=0
        for ep in range(self.info['total_episodes']):
            d=pq.read_table(root/f'data/chunk-000/episode_{ep:06}.parquet').to_pydict()
            n=len(d['frame_index']); assert d['frame_index']==list(range(n)) and set(d['episode_index'])=={ep}
            assert len(set(d['task_index']))==1 and set(d['task_index']) <= {0,1}
            tasks.extend(d['task_index'])
            state.extend(d['observation.state']);action.extend(d['action'])
            touch.extend(np.concatenate([d['observation.tactile.left'],d['observation.tactile.right']],1))
            anchors.extend(range(offset,offset+n-horizon+1));offset+=n
        self.state=np.asarray(state,np.float32);self.action=np.asarray(action,np.float32);self.touch=np.asarray(touch,np.float32)
        assert all(np.isfinite(a).all() for a in [self.state,self.action,self.touch])
        self.anchors=np.array(anchors);self.tasks=np.asarray(tasks);self.cache=CACHE/f'{split}.uint8'
        self.images=np.memmap(self.cache,mode='r',dtype=np.uint8,shape=(offset,188,336,3))
        if stats is None:
            assert split=='train';stats={}
            for key,a,floor in [('state',self.state,.01),('action',self.action,.01),('tactile',self.touch,1.)]:
                stats[key+'_mean']=a.mean(0);stats[key+'_std']=np.maximum(a.std(0),floor)
        self.stats=stats
    def __len__(self):return len(self.anchors)
    def __getitem__(self,i):
        a=self.anchors[i];s=self.stats
        state=np.concatenate([(self.state[a]-s['state_mean'])/s['state_std'],(self.touch[a]-s['tactile_mean'])/s['tactile_std']])
        action=(self.action[a:a+self.horizon]-s['action_mean'])/s['action_std']
        return {'task_index':torch.tensor(int(self.tasks[a])), 'observation.state':torch.from_numpy(state.copy()),'action':torch.from_numpy(action.copy()),
                'observation.image.0':torch.from_numpy(self.images[a].copy()).permute(2,0,1),
                'task':self.prompts[int(self.tasks[a])]}
