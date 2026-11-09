"""Auto-resuming supervisor for a continuation stage. Usage: stage_supervisor.py NAME"""
import subprocess, sys, time, math, json, shutil
from pathlib import Path
import torch
cfg=json.load(open(f'runs/{sys.argv[1]}.json'))
OUT=Path(cfg['output']); PARENT=Path(cfg['parent'])
START=int(torch.load(PARENT,map_location='cpu',weights_only=False)['step'])
TARGET=START+cfg['steps']; LR=cfg['lr']; MIN=cfg['min_lr']; WARM=cfg['warmup']
def lr_at(s):
    e=s-START
    if e<=WARM: return MIN+e/max(1,WARM)*(LR-MIN)
    p=(e-WARM)/(TARGET-START-WARM); return MIN+0.5*(LR-MIN)*(1+math.cos(math.pi*p))
if not OUT.exists(): shutil.copy2(PARENT, OUT); first=True
else: first=False
for attempt in range(50):
    step=int(torch.load(OUT,map_location='cpu',weights_only=False)['step'])
    if step>=TARGET: break
    cmd=['artemus-train',*cfg['args'],'--steps',str(TARGET),'--resume',str(OUT),'--output',str(OUT),
         '--allow-data-change','--override-resume-learning-rate','--lr-schedule','cosine','--min-learning-rate',str(MIN),'--device','cpu']
    if step==START: cmd+=['--learning-rate',str(LR),'--warmup-steps',str(WARM)]
    else: cmd+=['--learning-rate',f'{lr_at(step):.6g}','--warmup-steps','0']
    print('attempt',attempt,'step',step,flush=True)
    r=subprocess.run(cmd); print('exit',r.returncode,flush=True)
    if r.returncode==0: break
    time.sleep(5)
print('SUPERVISOR DONE',flush=True)
