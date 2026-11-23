"""Task B: attention-to-image curves via ONE teacher-forced forward pass per output.

Model loaded in bf16 with attn_implementation="eager" for the LLM. The module-level
eager_attention_forward of modeling_qwen3_vl is replaced (text-attention only) by a numerically
equivalent version that processes query rows in chunks (QK^T*scale + causal mask, fp32 softmax,
cast to bf16, @V -- same ops as eager) and, before discarding each chunk's probabilities,
reduces them to the attention mass on 4 key categories, averaged over heads:
  0 sink   : position 0 (<|im_start|>), a known attention sink
  1 image  : image placeholder tokens (config.image_token_id)
  2 prompt : all other prompt positions (chat template, vision_start/end, instruction text)
  3 gen    : previously generated tokens (incl. the current input token)
Stats for generated token i come from the query position that PREDICTS it (gen_start-1+i).
All 36 layers are recorded (cheap: probs @ onehot[L,4]). Vision tower runs with SDPA
(eager on 18k vision patches would need >20 GB per layer; it does not affect LLM attention).
Output: attn/<req_id>.npz with frac[layer, gen_idx, 4] (float16) + metadata; selection.csv.
"""
import os, sys, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

WORK = Path("/mnt/work/step1_loops_attention")
OUTD = WORK / "attn"; OUTD.mkdir(exist_ok=True)
MID = "Qwen/Qwen3-VL-8B-Instruct"
CAP = int(os.getenv("CAP", 15))
QCHUNK = 512

STATE = {"onehot": None, "q_lo": None, "store": None}
_orig_eager = mq.eager_attention_forward


def chunked_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    if not isinstance(module, mq.Qwen3VLTextAttention) or STATE["onehot"] is None:
        return _orig_eager(module, query, key, value, attention_mask, scaling, dropout, **kw)
    assert query.shape[0] == 1, "batch 1 only (no padding -> plain causal mask)"
    k = mq.repeat_kv(key, module.num_key_value_groups)
    v = mq.repeat_kv(value, module.num_key_value_groups)
    L = query.shape[2]; Lk = k.shape[2]
    out = torch.empty_like(query)
    onehot = STATE["onehot"]  # [Lk,4] fp32
    q_lo = STATE["q_lo"]
    stats = []
    kpos = torch.arange(Lk, device=query.device)
    for s in range(0, L, QCHUNK):
        e = min(L, s + QCHUNK)
        sc = torch.matmul(query[:, :, s:e], k.transpose(2, 3)) * scaling
        qpos = torch.arange(s, e, device=query.device)[:, None] + (Lk - L)
        sc = sc.masked_fill(kpos[None, :] > qpos, float("-inf"))
        p = torch.softmax(sc, dim=-1, dtype=torch.float32)
        if e > q_lo:
            a = max(s, q_lo)
            pr = p[0, :, a - s:e - s]                       # [H, q, Lk]
            stats.append((pr @ onehot).mean(0).half().cpu())  # [q,4]
        out[:, :, s:e] = torch.matmul(p.to(query.dtype), v)
        del sc, p
    STATE["store"][module.layer_idx] = torch.cat(stats, 0).numpy()
    return out.transpose(1, 2).contiguous(), None


mq.eager_attention_forward = chunked_eager


def select(df):
    loops = df[df.loop].sort_values("completion_tokens", ascending=False).head(CAP)
    ctrl_pool = df[~df.loop & (df.finish_reason == "stop")].copy()
    ctrls = []
    for _, r in loops.iterrows():
        pool = ctrl_pool[~ctrl_pool.req_id.isin(ctrls)]
        same = pool[pool.condition == r.condition]
        cand = same if len(same) else pool
        if not len(cand):
            break
        ctrls.append(cand.iloc[(cand.completion_tokens - r.completion_tokens).abs().argsort().iloc[0]].req_id)
    sel = pd.concat([loops.assign(role="loop"), df[df.req_id.isin(ctrls)].assign(role="control")])
    return sel


def main():
    df = pd.read_csv(WORK / "loop_metrics.csv")
    sel = select(df)
    sel.to_csv(WORK / "attn_selection.csv", index=False)
    print(sel[["req_id", "condition", "role", "completion_tokens", "loop_onset_token"]].to_string(), flush=True)
    raw = {}
    for line in open(WORK / "raw_outputs.jsonl"):
        d = json.loads(line)
        if d["req_id"] in set(sel.req_id) and "error" not in d:
            raw[d["req_id"]] = d

    proc = AutoProcessor.from_pretrained(MID)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16,
                                                                attn_implementation="eager", device_map="cuda:0")
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(MID, torch_dtype=torch.bfloat16,
                                                                attn_implementation="eager", device_map="cuda:0")
    try:
        model.model.visual.config._attn_implementation = "sdpa"
    except Exception as ex:
        print("could not switch vision to sdpa:", ex)
    model.eval()
    img_id = model.config.image_token_id
    meta = dict(transformers=transformers.__version__, torch=torch.__version__,
                text_attn=model.config.text_config._attn_implementation,
                vision_attn=getattr(model.model.visual.config, "_attn_implementation", "?"),
                deepstack_visual_indexes=list(model.config.vision_config.deepstack_visual_indexes),
                n_layers=model.config.text_config.num_hidden_layers, image_token_id=img_id)
    json.dump(meta, open(WORK / "attn_env.json", "w"), indent=1)
    print(meta, flush=True)

    for rid in sel.req_id:
        outp = OUTD / f"{rid}.npz"
        if outp.exists() or rid not in raw:
            continue
        d = raw[rid]
        t0 = time.time()
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(d["image"]).convert("RGB")},
                                             {"type": "text", "text": d["prompt"]}]}]
        enc = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt")
        p_ids = enc["input_ids"][0]
        gen = torch.tensor(d["gen_token_ids"], dtype=torch.long)
        ids = torch.cat([p_ids, gen])[None].cuda()
        P, N, L = len(p_ids), len(gen), len(p_ids) + len(gen)
        cat = torch.full((L,), 2, dtype=torch.long)
        cat[0] = 0
        cat[:P][p_ids == img_id] = 1
        cat[P:] = 3
        STATE["onehot"] = torch.nn.functional.one_hot(cat, 4).float().cuda()
        STATE["q_lo"] = P - 1
        STATE["store"] = {}
        with torch.no_grad():
            # transformers>=5 needs mm_token_type_ids for M-RoPE (0 text, 1 image)
            mm = (ids == img_id).long()
            o = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            pixel_values=enc["pixel_values"].cuda(), image_grid_thw=enc["image_grid_thw"].cuda(),
                            mm_token_type_ids=mm, use_cache=False)
            h = o.last_hidden_state[0, P - 1:L - 1]
            agree = 0
            for s in range(0, N, 512):
                agree += (model.lm_head(h[s:s + 512]).argmax(-1).cpu() == gen[s:s + 512]).sum().item()
        nl = meta["n_layers"]
        frac = np.stack([STATE["store"][l][:N] for l in range(nl)])  # [layers, N, 4]
        np.savez_compressed(outp, frac=frac, prompt_len=P, n_img=int((cat == 1).sum()), n_gen=N,
                            vllm_prompt_tokens=d["prompt_tokens"], argmax_agree=agree / max(N, 1))
        STATE["onehot"] = None
        print(f"{rid}: P={P} (vllm {d['prompt_tokens']}) img={int((cat == 1).sum())} N={N} "
              f"greedy-agree={agree / max(N, 1):.3f} {time.time() - t0:.0f}s "
              f"mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        del o, h; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
