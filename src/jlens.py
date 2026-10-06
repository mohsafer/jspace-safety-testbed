"""Minimal re-implementation of the Anthropic Jacobian lens (arXiv:2607.15495).

Definition (paper/README):
    lens_l(h) = unembed( J_l @ h ),   J_l = E[ dh_final / dh_l ]

Estimator (Hutchinson-style, matching "cotangents summed over target positions,
averaged over source positions"):

    For random cotangent c ~ N(0, I_d) applied at the final (post-final-norm)
    hidden state, autograd gives at layer l, source position s:

        g_l[s] = J_{s,s}^T c            (same-position variant, target window W=0)

    Then  E_c[ g_l[s] c^T ] = J_{s,s}^T, so averaging outer(g, c) over source
    positions, prompts, and cotangent samples estimates J_l^T.

The same-position variant (W=0) guarantees the last-layer lens reproduces the
model's own readout (the final LayerNorm Jacobian), which we use as a sanity
check. A target-window knob (sum over future positions) is kept to mirror the
paper's "current-and-future targets" option.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _final_norm(model):
    """Post-final-norm mapping for Llama/Qwen/SmolLM-style HF models.

    NOTE (transformers 5.x): `output_hidden_states` already returns the FINAL
    entry post-final-norm — logits == unembed @ hidden_states[-1] exactly
    (verified: cos = 1.0000 on SmolLM2-135M/-Instruct). So this helper must NOT
    be applied to hidden_states[-1]; it exists only for raw-path readouts.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "norm"):
        return inner.norm
    return None


@torch.enable_grad()
def fit_lens(
    model,
    prompts_ids: list[Tensor],
    n_cotangents: int = 8,
    target_window: int = 0,
    device: str = "cpu",
    show_progress: bool = True,
) -> list[Tensor]:
    """Fit per-layer lens matrices J_l. Returns list of [d, d] tensors (one per
    hidden-state index, i.e. n_layers+1 entries; entry 0 is the embedding layer).

    prompts_ids: list of 1-D LongTensors (variable length ok, one prompt each).
    """
    was_training = model.training
    model.eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    acc = [torch.zeros(d, d, device=device) for _ in range(n_layers + 1)]
    n_pos_total = 0

    iterator = range(len(prompts_ids))
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="fit lens", leave=False)

    for i in iterator:
        ids = prompts_ids[i].to(device)[None, :]          # [1, T]
        out = model(ids, output_hidden_states=True)
        hs = list(out.hidden_states)                      # n_layers+1 × [1, T, d]
        # transformers 5.x: hs[-1] is ALREADY post-final-norm (the lm_head
        # input) — use it directly as the lens target; do not norm again.
        hF = hs[-1]

        if target_window > 0:
            # sum cotangents over current-and-future targets: reuse one c per
            # position but backprop through the full graph (approximation of
            # the paper's all-future-targets variant).
            for k in range(n_cotangents):
                c = torch.randn_like(hF)
                loss = (hF * c).sum()
                grads = torch.autograd.grad(
                    loss, hs, retain_graph=(k < n_cotangents - 1)
                )
                for l, g in enumerate(grads):
                    # E over all (source,target) pairs: outer(g, c) summed
                    acc[l] += torch.einsum("btd,bte->de", g, c)
                n_pos_total += 1
        else:
            # same-position targets: per-position independent cotangents; ONE
            # backward per cotangent covers every source position.
            for k in range(n_cotangents):
                c = torch.randn_like(hF)
                loss = (hF * c).sum()
                grads = torch.autograd.grad(
                    loss, hs, retain_graph=(k < n_cotangents - 1)
                )
                for l, g in enumerate(grads):
                    acc[l] += torch.einsum("btd,bte->de", g, c)
            n_pos_total += ids.shape[1]

    # E_c[g c^T] = J^T  →  transpose back; average over positions & cotangents
    n = max(n_pos_total * (n_cotangents if target_window > 0 else n_cotangents), 1)
    # free each accumulator as its lens materializes — at 9B the double buffer
    # (acc + lenses, ~2.2 GB fp32) OOMs a 24 GB card that already holds the
    # bf16 model (failure: program-9 first attempt)
    lenses = []
    for l, m in enumerate(acc):
        J = (m / n).T.contiguous()
        if not torch.isfinite(J).all():
            raise FloatingPointError(f"lens layer {l} has non-finite entries")
        lenses.append(J)
        acc[l] = None
    del acc
    if was_training:
        model.train()
    return lenses


@torch.no_grad()
def lens_logits_at(
    model,
    lenses: list[Tensor],
    hidden_states: tuple[Tensor, ...],
    position: int = -1,
) -> list[Tensor]:
    """Per-layer readout logits: unembed(J_l @ h_l[pos]).

    Returns list of [vocab] logits, one per layer (entry 0 = embedding layer).
    The LAST entry is the model's own readout: hidden_states[-1] is already
    post-final-norm in transformers 5.x, so no norm is applied there.

    The math runs in float32 regardless of model dtype (fp16/bf16 models +
    NF4-quantized loads otherwise crash on mixed-dtype matmuls); casts are
    no-ops on the fp32 path.
    """
    unembed = model.get_output_embeddings().weight.float()      # [vocab, d]
    out = []
    for l, J in enumerate(lenses):
        h = hidden_states[l][0, position].float()               # [d]
        h_final = h if l == len(lenses) - 1 else J.float() @ h
        out.append(unembed @ h_final)
    return out


@torch.no_grad()
def direction_transport(
    lens: Tensor,
    unembed_weight: Tensor,
    direction: Tensor,
    n_null: int = 32,
    seed: int = 0,
) -> float:
    """Accessibility amplification of a latent direction under ONE layer's lens.

    A = || U J v || / median_m || U J v_m ||  with v_m random unit vectors.
    A >> 1 ⇒ the direction is strongly transported toward output space
    (behaviorally accessible, i.e. lives in J-space); A ~ 1 ⇒ latent-only.

    Runs on the device of `lens` (CUDA-safe: generator + nulls follow it).
    The U-projection is computed in VOCAB CHUNKS with the squared norms
    accumulated in fp64 — mathematically the same value as the monolithic
    fp64 matmul, but it never materializes U in fp64 (gemma-2's 256k vocab
    in fp64 is ~5 GB and OOMs a 24 GB card that holds the bf16 model;
    failure: program-9 first attempt). Chunked matvec runs in the unembed's
    own dtype (bf16 at 9B ⇒ ~0.5% per-element error, common to base and
    nulls ⇒ the RATIO is preserved to well under the reported precision).
    """
    dev = lens.device
    g = torch.Generator(device=dev).manual_seed(seed)
    d = direction.shape[0]
    lens = lens.double()
    v = direction.to(dev).double()

    C = 65536  # vocab chunk (gemma 256128 -> 4 chunks)

    def sqnorm_Ux(x: Tensor) -> float:
        xb = x.to(unembed_weight.dtype)
        tot = 0.0
        for i in range(0, unembed_weight.shape[0], C):
            r = unembed_weight[i:i + C] @ xb
            tot += float(r.double() @ r.double())
        return tot

    base = sqnorm_Ux(lens @ v) ** 0.5
    nulls = []
    for _ in range(n_null):
        nv = torch.randn(d, generator=g, device=dev).double()
        nv = nv / nv.norm()
        nulls.append(sqnorm_Ux(lens @ nv) ** 0.5)
    med = torch.tensor(nulls, device=dev).median().clamp_min(1e-12)
    return float(base / med)
