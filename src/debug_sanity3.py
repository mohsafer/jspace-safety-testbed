"""Trace the instruct-only NaN in direction transport: check H finiteness per prompt/layer."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .jlens import fit_lens, direction_transport
from .prompts import HARMFUL, BENIGN, CALIBRATION_PROMPTS
from .run_smoke import prep_ids

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

ids_list, texts, labels = [], [], []
for lab, ps in (("harmful", HARMFUL), ("benign", BENIGN)):
    for p in ps:
        ids_list.append(prep_ids(tok, p, True))
        texts.append(p)
        labels.append(1 if lab == "harmful" else 0)

H = []
bad = []
for i, (ids, p) in enumerate(zip(ids_list, texts)):
    with torch.no_grad():
        out = model(ids[None], output_hidden_states=True)
    hs = torch.stack([h[0, -1] for h in out.hidden_states])
    if not torch.isfinite(hs).all():
        bad.append((i, p, [l for l, h in enumerate(out.hidden_states)
                           if not torch.isfinite(h).all()]))
    H.append(hs)
print("prompts with non-finite hidden states:", bad if bad else "none")
H = torch.stack(H)
labels = torch.tensor(labels)
print("H shape:", H.shape, "finite:", bool(torch.isfinite(H).all()))

calib = [prep_ids(tok, p, True) for p in CALIBRATION_PROMPTS]
lenses = fit_lens(model, [c for c in calib if c.shape[0] >= 4],
                  n_cotangents=8, show_progress=False)
print("lenses finite:", all(bool(torch.isfinite(J).all()) for J in lenses))

unembed = model.get_output_embeddings().weight
for l in range(H.shape[1]):
    v = (H[labels == 1, l].mean(0) - H[labels == 0, l].mean(0))
    vn = v.norm().item()
    vn_ = v / (v.norm() if v.norm() > 0 else torch.tensor(1.0))
    a = direction_transport(lenses[l], unembed, vn_)
    if a != a or vn != vn:
        print(f"L{l}: v.norm={vn} amp={a} "
              f"v nan={bool(torch.isnan(v).any())} "
              f"lens nan={bool(torch.isnan(lenses[l]).any())}")
print("done")
