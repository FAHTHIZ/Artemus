ArtemisLM S2 task SFT checkpoint (Experiment 0014), continued from S1 (`s1-aya-6m-step13300`).

- 6,451,200 parameters; same architecture as F0/F1/S1
- +1,500 steps of response-only SFT on a mix of human-written data: OASST1 English conversations (Apache-2.0), GSM8K official train (MIT; 6,011 examples, held-out hash buckets excluded) and MBPP train task ids 601-974 (CC-BY-4.0). No benchmark test split was used. LR 2e-4 -> 2e-5 cosine, seed 2029
- Capability validation loss 3.252 (step 13500) -> **3.102** (step 14800); train 2.925
- Held-out bits/byte: FineWeb-Edu 1.410, FineMath 1.632 (S1: 1.479 / 1.692; F1 base: 1.350 / 1.508)
- Internal held-out capability diagnostic (greedy, `capability-eval.json`): GSM8K-style math 0/100 correct (answers parsed for 12/100, F1 parsed 5/100); MBPP-style Python 0/38 syntax-valid (F1 6/38). These are not official benchmark scores
- Prompt format: `<|prompter|>\n{question}\n<|assistant|>\n` ; answers end with `<|end|>`
- Uses GSM8K `<<a+b=c>>` calculator-annotation format but the arithmetic is wrong. At 6.45M parameters this is a research artefact, not an assistant.
