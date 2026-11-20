import hashlib,json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from data import SH5Dataset,ROOT,CACHE
from model import ROOT as REPO
receipts={};fingerprints={};source_sets=[]
train=SH5Dataset('train');val=SH5Dataset('validation',train.stats)
for split,ds in [('train',train),('validation',val)]:
    root=ROOT/split
    mapping=json.loads((root/'conversion_receipt.json').read_text())['source_episode_mapping']
    source_sets.append({x['source'] for x in mapping});assert len(source_sets[-1])==len(mapping)
    offset=0;errors=[];task_episodes={0:0,1:0}
    for ep in range(ds.info['total_episodes']):
        p=root/f'data/chunk-000/episode_{ep:06}.parquet';n=pq.read_metadata(p).num_rows
        fingerprints[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        tid=int(ds.tasks[offset]);task_episodes[tid]+=1
        raw=json.loads((Path(mapping[ep]['source'])/'episode_info.json').read_text())
        assert raw['task_instruction']==ds.source_prompts[tid],(ep,raw['task_instruction'])
        if ep in [0,77,78,ds.info['total_episodes']//2,ds.info['total_episodes']-1]:
            v=root/f'videos/chunk-000/observation.images.rgb.cam_left_head/episode_{ep:06}.mp4'
            cap=cv2.VideoCapture(str(v));frame=n//2;cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release();assert ok
            image=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(336,188),interpolation=cv2.INTER_LINEAR)
            err=int(np.abs(image.astype(np.int16)-ds.images[offset+frame].astype(np.int16)).max());assert err==0;errors.append(err)
        anchors=ds.anchors[(ds.anchors>=offset)&(ds.anchors<offset+n)]
        assert (anchors+49<offset+n).all();offset+=n
    for name in ['meta/info.json','meta/tasks.jsonl','conversion_receipt.json']:
        p=root/name;fingerprints[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    receipts[split]={'episodes':ds.info['total_episodes'],'episodes_by_task':task_episodes,'frames':offset,'windows':len(ds),'representative_decode_errors':errors,'frames_by_task':{t:int((ds.tasks==t).sum()) for t in [0,1]}}
assert not source_sets[0]&source_sets[1]
receipts.update(source_disjoint=True,parquet_metadata_sha256_count=len(fingerprints),all_raw_episode_prompts_verified=True,source_prompts=train.source_prompts,prompts=train.prompts,overlay=False,target='raw absolute joint radians; no clipping',camera='cam_left_head RGB resized 336x188 then Rho aspect-preserving 256x256',tactile='left45,right45; combined train-only mean/std floor1')
p=REPO/'work/sh5/provenance';(p/'data_validation_mixed.json').write_text(json.dumps(receipts,indent=2));(p/'dataset_fingerprint_mixed.json').write_text(json.dumps(fingerprints,indent=2));print(json.dumps(receipts))
