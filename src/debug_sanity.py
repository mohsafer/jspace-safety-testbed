"""Debug the two smoke anomalies: base-model sanity cos=0.05, instruct amps=NaN."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

for name in ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"]:
    print("=" * 60)
    print(name)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    print("class:", type(model).__name__)
    print("has model.model:", hasattr(model, "model"),
          "| model.model has norm:", hasattr(getattr(model, "model", None), "norm"),
          "| tie_word_embeddings:", model.config.tie_word_embeddings)
    ids = tok("The history of banking is old.", return_tensors="pt").input_ids
    out = model(ids, output_hidden_states=True)
    U_out = model.get_output_embeddings().weight
    U_head = model.lm_head.weight
    U_emb = model.model.embed_tokens.weight
    n = model.model.norm
    hF = n(out.hidden_states[-1])
    manual = U_out @ hF[0, -1]
    cos = torch.nn.functional.cosine_similarity(out.logits[0, -1], manual, dim=0).item()
    print(f"logits nan={torch.isnan(out.logits).any().item()} "
          f"absmax={out.logits.abs().max().item():.3e}")
    print(f"U_out is U_head: {U_out is U_head} | U_out == U_emb: {torch.equal(U_out, U_emb)}")
    print(f"U nan={torch.isnan(U_out).any().item()} absmax={U_out.abs().max().item():.3e}")
    print(f"norm.weight nan={torch.isnan(n.weight).any().item()}")
    print(f"cos(logits, U_out@norm(h)) = {cos:.4f}")
    hs_nans = [torch.isnan(h).any().item() for h in out.hidden_states]
    print("hidden_states nan per layer:", any(hs_nans), hs_nans[:3])
