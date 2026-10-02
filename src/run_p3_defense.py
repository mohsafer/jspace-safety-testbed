"""R6 / H4 second half: J-space-LAT defense (runbook Phase 7).

H4 as planned includes "J-space-LAT closes the gap" — the only P3 half not run
on the P100. Design (research/06 R6, arXiv:2407.15549 analogy): CONTINUE DPO
training on the target with an added term that RAISES the monitor reading on
adversarial prompts — the mirror image of run_dpo's harm_dcg_penalty, which
LOWERS harm-axis accessibility during tuning:

    loss = DPO_loss(HH pairs) + lam_def * soft_saf(attacked prompts)

where soft_saf is the differentiable safety-axis lexicon mass of the lens
readout at the last attacked-prompt position (frozen lens, same surrogate the
silent-arm attack tries to SUPPRESS). The adversarial batch is refreshed each
round by re-attacking the CURRENT model with a reduced GCG budget (round 0
uses the stored attacked suffixes). Post-condition: re-run the attack suite
against the defended model (run_p3_gcg --target-path ...) and report
defended/undefended deltas per arm.

Artifacts: artifacts/defense/<defslug>/{merged/, defense_log.json}.
Run:  .venv/bin/python -m src.run_p3_defense --target smol135dpo --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ensure_tuning_pairs
from .lexicons import build_token_hits
from .run_dpo import pair_tensors, sequence_logprob, sequence_logprob_ref
from .run_p3_gcg import (SAFETY_AI, TARGETS, _disc, soft_saf_mass)
from .run_smoke import prep_ids

ROOT = Path(__file__).resolve().parent.parent
P3 = ROOT / "artifacts" / "p3"
OUT = ROOT / "artifacts" / "defense"


def load_adversarial(slug: str, device: str):
    """Stored attacked suffixes for one target, both arms -> list of
    (prompt, suffix_ids)."""
    rows = []
    for arm in ("vanilla", "silent"):
        for pj in sorted((P3 / slug / arm).glob("prompt*.json")):
            r = json.loads(pj.read_text())
            if "suffix_ids" in r:
                rows.append({"prompt": r["prompt"], "suffix_ids": r["suffix_ids"]})
    return rows


def soft_saf_batch(model, lenses, tok, tok_hits_soft, rows, device, temp=1.0):
    """Mean differentiable safety-axis mass over the attacked batch (the term
    the defense RAISES and the silent attack SUPPRESSES)."""
    totals = []
    for row in rows:
        if getattr(tok, "chat_template", None) and row.get("is_instruct", False):
            pre = tok.apply_chat_template(
                [{"role": "user", "content": row["prompt"] + " "}],
                tokenize=False, add_generation_prompt=True)
            ids_p = tok(pre, return_tensors="pt",
                        add_special_tokens=False).input_ids[0][:-1]
        else:
            ids_p = tok(row["prompt"] + " ", return_tensors="pt").input_ids[0]
        ids = torch.cat([ids_p, torch.tensor(row["suffix_ids"])]).to(device)
        out = model(ids[None], output_hidden_states=True)
        totals.append(soft_saf_mass(model, lenses, out.hidden_states,
                                    ids.shape[0] - 1, tok_hits_soft, temp))
    return torch.stack(totals).mean()


def reattack_current(model, tok, lenses, tok_hits, disc, args, device):
    """Refresh the adversarial batch: silent-arm GCG vs the CURRENT weights
    (reduced budget). Reuses run_p3_gcg's machinery via a local import to
    avoid a circular dependency at module load.
    NOTE: NOT @torch.no_grad() — the GCG search differentiates through the
    model (its backward needs a live graph); measurement helpers inside
    attack_prompt carry their own no_grad scopes."""
    from .run_p3_gcg import attack_prompt, calibrate_harm_directions
    from .data import ensure_eval_sets
    ev = ensure_eval_sets()
    rng = random.Random(args.seed + 1000)
    danger = [p for _, p in ev["danger"]]
    prompts = rng.sample(danger, args.reattack_prompts)
    harm_dirs, harm_stats = calibrate_harm_directions(
        model, tok, args.is_instruct, device)
    cal = json.loads((P3 / args.slug / "monitor_calibration.json").read_text())
    tau95, b_med, b_sd = (cal["tau95"], cal["benign_median"], cal["benign_sd"])
    tok_hits_soft = tok_hits[:, SAFETY_AI].float()
    rows = []
    for pi, prompt in enumerate(prompts):
        if args.is_instruct and getattr(tok, "chat_template", None):
            pre = tok.apply_chat_template(
                [{"role": "user", "content": prompt + " "}],
                tokenize=False, add_generation_prompt=True)
            ids_p = tok(pre, return_tensors="pt",
                        add_special_tokens=False).input_ids[0][:-1]
        else:
            ids_p = tok(prompt + " ", return_tensors="pt").input_ids[0]
        affirm = "Sure, here is"
        r = attack_prompt(model, tok, lenses, tok_hits, tok_hits_soft, disc,
                          harm_dirs, harm_stats, argparse.Namespace(
                              **{**vars(args), "arm": "silent",
                                 "batch": args.gcg_batch}),
                          ids_p.cpu(), affirm, tau95, b_med, b_sd, device)
        rows.append({"prompt": prompt, "suffix_ids": r["suffix_ids"],
                     "is_instruct": args.is_instruct})
        print(f"    re-attack {pi+1}/{len(prompts)}: success={r['affirm_success']} "
              f"T={r['T_saf_final']:.2f}", flush=True)
    return rows


def train_defense(args):
    name, is_instruct = TARGETS[args.target]
    if name.startswith("dpo:"):
        base_slug = name[4:]
        name = str(ROOT / "artifacts" / "dpo" / base_slug / "merged")
    else:
        base_slug = name.split("/")[-1]
    defslug = f"{base_slug}_defended_lam{args.lam_def}"
    out_dir = OUT / defslug
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "merged").exists() and not args.force:
        print("already defended:", out_dir)
        return
    args.slug = base_slug
    args.is_instruct = is_instruct

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    from peft import LoraConfig, get_peft_model
    lconf = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                       lora_dropout=0.05,
                       target_modules=["q_proj", "v_proj"], bias="none",
                       task_type="CAUSAL_LM")
    model = get_peft_model(model, lconf).to(args.device)

    # frozen monitor lens (the attack's own cached lens for this target)
    lenses = [J.to(args.device) for J in torch.load(
        P3 / f"lens_{base_slug}.pt", weights_only=True)]
    tok_hits = build_token_hits(
        tok, vocab_size=model.config.vocab_size).to(args.device)
    tok_hits_soft = tok_hits[:, SAFETY_AI].float()
    disc = _disc(args.k_metric, torch.device(args.device))

    adv_rows = load_adversarial(base_slug, args.device)
    print(f"{len(adv_rows)} stored adversarial rows (round 0)")

    pairs = ensure_tuning_pairs(args.n_pairs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    log = []
    t0 = time.time()
    step = 0
    for round_i in range(args.rounds):
        if round_i > 0:
            print(f"== round {round_i}: re-attacking current model", flush=True)
            adv_rows = reattack_current(model, tok, lenses, tok_hits, disc,
                                        args, args.device)
        order = rng.sample(range(len(pairs)), min(args.steps_per_round,
                                                  len(pairs)))
        for bi in range(0, len(order), args.batch):
            batch = [pairs[j] for j in order[bi:bi + args.batch]]
            pol_c, pol_r, ref_c, ref_r = [], [], [], []
            for ex in batch:
                ids_c, np_c = pair_tensors(tok, ex["prompt"], ex["chosen"],
                                           args.max_len)
                ids_r, np_r = pair_tensors(tok, ex["prompt"], ex["rejected"],
                                           args.max_len)
                pol_c.append(sequence_logprob(model, ids_c.to(args.device), np_c)[0])
                pol_r.append(sequence_logprob(model, ids_r.to(args.device), np_r)[0])
                ref_c.append(sequence_logprob_ref(model, ids_c.to(args.device), np_c))
                ref_r.append(sequence_logprob_ref(model, ids_r.to(args.device), np_r))
            pol_c, pol_r = torch.stack(pol_c), torch.stack(pol_r)
            ref_c = torch.tensor(ref_c, device=args.device)
            ref_r = torch.tensor(ref_r, device=args.device)
            logits_dpo = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))
            loss_dpo = -F.logsigmoid(logits_dpo).mean()
            adv_batch = rng.sample(adv_rows, min(args.adv_per_batch, len(adv_rows)))
            pen = soft_saf_batch(model, lenses, tok, tok_hits_soft, adv_batch,
                                 args.device)
            loss = loss_dpo - args.lam_def * pen     # RAISE the monitor reading
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            log.append({"round": round_i, "step": step, "loss": float(loss),
                        "loss_dpo": float(loss_dpo), "saf_mass": float(pen)})
            if step % 5 == 0 or step == 1:
                print(f"r{round_i} step {step} dpo={float(loss_dpo):.4f} "
                      f"saf={float(pen):.4f} ({time.time()-t0:.0f}s)", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir / "merged")
    tok.save_pretrained(out_dir / "merged")
    (out_dir / "defense_log.json").write_text(json.dumps(log, indent=1))
    (out_dir / "train_report.json").write_text(json.dumps({
        "target": args.target, "rounds": args.rounds, "lam_def": args.lam_def,
        "steps": step, "lr": args.lr}, indent=1))
    print("saved:", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="smol135dpo", choices=list(TARGETS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps-per-round", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--adv-per-batch", type=int, default=4)
    ap.add_argument("--gcg-batch", type=int, default=64,
                    help="candidate batch for the round refresh re-attacks")
    ap.add_argument("--lam-def", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-pairs", type=int, default=3000)
    ap.add_argument("--k-metric", type=int, default=100)
    ap.add_argument("--reattack-prompts", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=25, help="(re-attack budget)")
    ap.add_argument("--suffix-len", type=int, default=20)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mu", type=float, default=10.0)
    ap.add_argument("--soft-temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    train_defense(args)


if __name__ == "__main__":
    main()
