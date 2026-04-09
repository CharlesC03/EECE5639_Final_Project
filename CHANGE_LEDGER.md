# Change Ledger

Changes made by Claude, in reverse chronological order.

---

### 2026-04-09
**Prompt:** "Shouldn't I use a device? Not specific to cuda?"
**Summary:** Updated `plonk/models/samplers/custom_sampler.py:78` — replaced hardcoded `'cuda'` in `torch.amp.autocast` with `x.device.type` to be device-agnostic (works on CPU, CUDA, MPS, etc.).
