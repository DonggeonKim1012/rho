"""48-hour SH5 Rho trainer; bounded rolling full checkpoints and online scalar logging."""
import argparse, fcntl, json, math, os, random, shutil, signal, time, traceback, uuid
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data import SH5Dataset, ROOT as DATA_ROOT
from model import build, ROOT

def atomic_json(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,default=str)+'\n');tmp.replace(p)
def move(b):return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',type=int,required=True);ap.add_argument('--probe',action='store_true');ap.add_argument('--steps',type=int,default=10);ap.add_argument('--hours',type=float,default=48);ap.add_argument('--resume',action='store_true');ap.add_argument('--save-every',type=int,default=30000);args=ap.parse_args()
    if args.save_every<1:raise ValueError('save-every must be positive')
    out=ROOT/'work/sh5'/('probe_'+str(args.batch) if args.probe else 'run');out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.probe and not args.resume and (out/'config.json').exists():raise RuntimeError('Existing run requires --resume')
    torch.set_num_threads(4);random.seed(905);np.random.seed(905);torch.manual_seed(905);torch.cuda.manual_seed_all(905)
    torch.backends.cuda.matmul.allow_tf32=True
    td=SH5Dataset('train');vd=SH5Dataset('validation',td.stats)
    contract=json.loads((ROOT/'sh5/contract.json').read_text())
    np.savez(out/'normalization.npz',**td.stats)
    policy,replaced=build();params=[p for p in policy.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    loader=DataLoader(td,batch_size=args.batch,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    validation_indices=[]
    for task in (0,1):
        candidates=np.flatnonzero(vd.tasks[vd.anchors]==task)
        validation_indices.extend(candidates[np.linspace(0,len(candidates)-1,64,dtype=int)].tolist())
    vl=DataLoader(Subset(vd,validation_indices),batch_size=min(args.batch,8),shuffle=False,num_workers=0)
    started=time.time();step=0;run=None
    def load(p):
        nonlocal step,started
        ck=torch.load(p,map_location='cpu',weights_only=False)
        if ck.get('prompts')!=td.prompts:raise ValueError('Checkpoint language contract differs; start a new run')
        policy.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);step=ck['step'];started=ck['started'];torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);del ck
    if args.resume:
        load(out/'latest.pt')
        print(f'RESUMED step={step} original_started={started} save_every={args.save_every} keep=1',flush=True)
        atomic_json(out/'resume_verification.json',{'step':step,'original_started_epoch':started,'optimizer_state_count':len(opt.state),'prompts':td.prompts,'save_every':args.save_every,'keep_checkpoints':1})
    deadline=started+args.hours*3600
    if not args.probe:
        import wandb
        identity=out/'wandb_identity.json'
        if identity.exists():ident=json.loads(identity.read_text())
        else:
            ident=dict(entity='donggunkim-kyung-hee-university',project='sh5-rho-tactile',id=uuid.uuid4().hex[:8],name='task905-943-rho-tactile-pad-prompt-48h');atomic_json(identity,ident)
        run=wandb.init(**ident,resume='allow',config={'batch':args.batch,'hours':args.hours,'state':54,'tactile':90,'action':54,'horizon':50,'fps':30,'base':'microsoft/rho-base','train_expert_only':True,'data':str(DATA_ROOT),'loss':'flow matching MSE','lr':1e-4,'prompts':td.prompts,'overlay':False,'save_every':args.save_every,'keep_checkpoints':1},dir=str(out))
        atomic_json(out/'wandb.json',{'url':run.url})
    atomic_json(out/'config.json',{'batch':args.batch,'save_every':args.save_every,'keep_checkpoints':1,'final_save':True,'prompts':td.prompts,'source_prompts':td.source_prompts,'contract':contract,'deadline_epoch':deadline,'started_epoch':started,'data':str(DATA_ROOT),'reinitialized_keys':replaced,'trainable_parameters':sum(p.numel() for p in params),'policy':str(policy.config)})
    atomic_json(out/'status.json',{'state':'running','step':step,'pid':os.getpid(),'deadline_epoch':deadline,'phase':'resumed' if args.resume else 'initializing'})
    stop=False
    def handle(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
    def save():
        tmp=out/'checkpoint.tmp';latest=out/'latest.pt'
        if shutil.disk_usage(out).free < 25*2**30:raise RuntimeError('Insufficient free space for atomic full checkpoint')
        torch.save({'model':policy.state_dict(),'optimizer':opt.state_dict(),'step':step,'started':started,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'stats':td.stats,'prompts':td.prompts,'contract':contract,'config':policy.config,'batch':args.batch},tmp)
        # Publish only a complete new checkpoint; retain one completed file.
        tmp.replace(latest)
        atomic_json(out/'checkpoint.json',{'step':step,'bytes':latest.stat().st_size,'path':str(latest),'saved_epoch':time.time()})
    def evaluate():
        policy.eval();losses=[];task_losses={0:[],1:[]}
        with torch.no_grad():
            for i,b in enumerate(vl):
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=policy(move(b))
                assert torch.isfinite(loss);losses.append(loss.item())
                task=0 if 'right hand' in b['task'][0] else 1
                assert len(set(b['task']))==1
                task_losses[task].append(loss.item())

        policy.train();policy.model.flow_model.vlm_backbone.eval()
        if run:run.log({f'validation/task{905 if t==0 else 943}_flow_mse':float(np.mean(values)) for t,values in task_losses.items()},step=step)
        return float(np.mean(losses))
    torch.cuda.reset_peak_memory_stats();last_val=time.time();it=iter(loader);t0=time.time()
    policy.train();policy.model.flow_model.vlm_backbone.eval()
    while not stop and (step<args.steps if args.probe else time.time()<deadline):
        try:b=next(it)
        except StopIteration:it=iter(loader);b=next(it)
        opt.zero_grad(set_to_none=True)
        progress=min(1.,(time.time()-started)/(args.hours*3600));lr=1e-4*min(1.,(step+1)/1000)*(.1+.9*(1+math.cos(math.pi*progress))/2)
        for group in opt.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=policy(move(b))
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward()
        tactile=policy.model.flow_model.state_projector.tactile_encoder
        tactile_grad=sum(p.grad.float().norm().item() for p in tactile.parameters() if p.grad is not None)
        if not math.isfinite(tactile_grad) or tactile_grad<=0:raise RuntimeError('tactile encoder gradient missing/nonfinite')
        grad=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        tracked=next(tactile.parameters());before=tracked.detach().clone() if step==0 else None
        opt.step()
        if before is not None:
            delta=(tracked.detach()-before).abs().max().item()
            if delta<=0:raise RuntimeError('tactile weights did not update')
            atomic_json(out/'tactile_update.json',{'max_abs_update':delta,'dtype':str(tracked.dtype)})
        step+=1
        if args.probe or step%10==0 or step==1:
            torch.cuda.synchronize();record={**metrics,'step':step,'loss':loss.item(),'lr':lr,'grad_norm':float(grad),'tactile_grad_norm':tactile_grad,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'elapsed_seconds':time.time()-t0,'state':'running','pid':os.getpid(),'deadline_epoch':deadline}
            print(json.dumps(record),flush=True);atomic_json(out/'status.json',record)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if run:run.log({k:v for k,v in record.items() if isinstance(v,(int,float))},step=step)
        if not args.probe and (step==10 or time.time()-last_val>=1800):
            val=evaluate();run.log({'validation/flow_mse_128_spread_windows':val},step=step);last_val=time.time()
        if not args.probe and step % args.save_every == 0:save()
    torch.cuda.synchronize()
    result={'state':'completed','step':step,'batch':args.batch,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'free_gib':torch.cuda.mem_get_info()[0]/2**30,'seconds_per_step':(time.time()-t0)/max(1,step),'reason':'probe' if args.probe else ('signal' if stop else 'deadline')}
    if not args.probe or args.batch==1:
        result['validation_loss']=evaluate();save()
        if args.probe:
            load(out/'latest.pt')
            b=move(next(iter(vl)));policy.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=policy.sample_actions(b)['actions']
            assert pred.shape==(len(b['action']),50,54) and torch.isfinite(pred).all()
            # Prove restored optimizer can perform another update.
            policy.train();policy.model.flow_model.vlm_backbone.eval();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):l,_=policy(b)
            l.backward();opt.step();result['save_reload_inference_resume_pass']=True
    atomic_json(out/'completion.json',result);atomic_json(out/'status.json',result);print(json.dumps(result),flush=True)
    if run and 'validation_loss' in result:run.log({'validation/final_flow_mse':result['validation_loss']},step=step)
    if run:run.finish()
if __name__=='__main__':main()
