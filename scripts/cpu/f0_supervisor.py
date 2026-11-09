"""Auto-resuming supervisor for the CPU F0 foundation run."""
import subprocess, sys, time, math
from pathlib import Path
import torch
TARGET=10000; LR=2e-3; MIN=2e-4; WARM=400
OUT=Path('ckpt/artemus-f0-edu-6m.pt')
base=['artemus-train','--data','data/edu-hq/train.jsonl','--validation-data','data/edu-hq/validation.jsonl',
 '--document-format','jsonl','--token-cache-dir','data/tokcache','--data-manifest','data/edu-hq/manifest.json',
 '--phase','foundation_pretraining','--tokenizer','sentencepiece','--tokenizer-model','data/artemus-edu-8k.model',
 '--steps',str(TARGET),'--batch-size','16','--block-size','256','--layers','6','--heads','4','--kv-heads','2',
 '--embedding-size','256','--use-rope','--use-qk-norm','--use-swiglu','--lr-schedule','cosine',
 '--min-learning-rate',str(MIN),'--eval-interval','500','--checkpoint-interval','250','--eval-iters','20',
 '--output',str(OUT),'--device','cpu','--seed','2026']
def lr_at(s):
    if s<=WARM: return MIN+s/WARM*(LR-MIN)
    p=(s-WARM)/(TARGET-WARM); return MIN+0.5*(LR-MIN)*(1+math.cos(math.pi*p))
for attempt in range(50):
    cmd=list(base)
    if OUT.exists():
        step=int(torch.load(OUT,map_location='cpu',weights_only=False)['step'])
        if step>=TARGET: break
        # Continue the same global cosine curve: decay from lr(step) to MIN over the remainder.
        cmd+=['--resume',str(OUT),'--learning-rate',f'{lr_at(step):.6g}','--warmup-steps','0','--override-resume-learning-rate']
        print(f'resume attempt={attempt} step={step} lr={lr_at(step):.6g}',flush=True)
    else:
        cmd+=['--learning-rate',str(LR),'--warmup-steps',str(WARM)]
    r=subprocess.run(cmd)
    print('exit',r.returncode,flush=True)
    if r.returncode==0: break
    time.sleep(5)
print('SUPERVISOR DONE',flush=True)
