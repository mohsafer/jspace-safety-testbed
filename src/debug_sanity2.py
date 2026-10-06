"""Find the exact final-path relation in transformers 5.17 + reproduce smoke anomalies."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .jlens import fit_lens, lens_logits_at, _final_norm, direction_transport

cos = torch.nn.functional.cosine_similarity

for name, is_instr in [("HuggingFaceTB/SmolLM2-135M", False),
                       ("HuggingFaceTB/SmolLM2-135M-Instruct", True)]:
    print("=" * 70)
    print(name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    text = ("The history of banking can be traced back to ancient merchant temples.")
    if is_instr and getattr(tok, "chat_template", None):
        s = tok.apply_chat_template([{"role": "user", "content": text}],
                                    tokenize=False, add_generation_prompt=True)
        ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids
    else:
        ids = tok(text, return_tensors="pt").input_ids
    print("chat-template prompt:", is_instr, "| last tokens:",
          tok.convert_ids_to_tokens(ids[0, -3:].tolist()))
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        h_last = out.hidden_states[-1][0, -1]
        U = model.get_output_embeddings().weight
        n = model.model.norm
        print(f"rms(hs[-1][-1]) = {h_last.pow(2).mean().sqrt().item():.3f}  "
              f"(≈1 ⇒ post-norm, large ⇒ pre-norm)")
        print(f"cos(logits, U@h)        = {cos(out.logits[0,-1], U@h_last, dim=0).item():.4f}")
        print(f"cos(logits, U@norm(h))  = {cos(out.logits[0,-1], U@(n(h_last[None])[0]), dim=0).item():.4f}")
        # exact smoke sanity path
        lens = [torch.eye(model.config.hidden_size) for _ in range(model.config.num_hidden_layers + 1)]
        ll = lens_logits_at(model, lens, out.hidden_states, -1)
        print(f"cos(logits, lens_logits_at[-1]) [identity lens] = "
              f"{cos(out.logits[0,-1], ll[-1], dim=0).item():.4f}")

    # fit a small lens, then check finiteness + NaN origin in amps
    calib = [ids[0]] + [tok("Ocean currents regulate the global climate by "
                            "transporting heat.", return_tensors="pt").input_ids[0]]
    lenses = fit_lens(model, calib, n_cotangents=4, show_progress=False)
    print("lens finiteness per layer:",
          [bool(torch.isfinite(J).all().item()) for J in lenses])
    print("lens absmax per layer:", [f"{J.abs().max().item():.2e}" for J in lenses])
    with torch.no_grad():
        out2 = model(ids, output_hidden_states=True)
        H = torch.stack([h[0, -1] for h in out2.hidden_states])
    v = H[0] - H[1]  # fake direction diff
    v = v / v.norm()
    unembed = model.get_output_embeddings().weight
    amps = []
    for l in range(len(lenses)):
        a = direction_transport(lenses[l], unembed, v)
        amps.append(a)
        if a != a:  # NaN check
            J = lenses[l]
            t1 = unembed @ (J @ v)
            print(f"  L{l}: amp NaN | J finite={bool(torch.isfinite(J).all())} "
                  f"J absmax={J.abs().max().item():.3e} "
                  f"U@Jv absmax={t1.abs().max().item():.3e} "
                  f"norm={t1.norm().item():.3e}")
    print("amps:", [f"{a:.2f}" for a in amps])
