ArtemisLM S1 English instruction-tuned checkpoint (Experiment 0013), continued from F1 (`f1-math-6m-step12500`).

- 6,451,200 parameters; same architecture as F0/F1
- +800 steps of response-only SFT (loss only on assistant tokens) on 3,775 English Aya Dataset conversations (human-written, Apache-2.0), LR 3e-4 -> 3e-5 cosine
- Aya validation loss 3.496 -> **3.314** (train 2.515: ~3.4 epochs, mild overfitting)
- Prompt format: `<|prompter|>\n{question}\n<|assistant|>\n` ; the model ends answers with `<|end|>`
- Follows the chat format and answers very simple facts ("The capital of France is Paris."), but most content is wrong or incoherent. Research artefact, not an assistant.

Data attribution: Aya Dataset by Cohere For AI (https://huggingface.co/datasets/CohereLabs/aya_dataset), Apache-2.0; English (`eng`) rows only.
