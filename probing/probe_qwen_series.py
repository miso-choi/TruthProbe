#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probe_unified.py — Head-level probing (LLM & VLM), 이미지(PHD) 모드 토글 (--img)

모드
- LLM: 순수 텍스트(C+Q+A), 마지막 토큰에서 self_attn.o_proj 입력 캡처
- VLM (text_only): 이미지 없이 텍스트만
- VLM (black): 검은 더미 이미지를 넣어 비전 경로 활성화
- VLM (--img): PHD 스타일 이미지+Q+A jsonl에서 이미지 경로를 읽어 실제 이미지 사용

입력
- --jsonl_path: LLM/VLM(text_only/black)에서는 C+Q+A(jsonl: context/question/answer/label)
               --img 가 켜진 경우에는 PHD 이미지 포맷(jsonl: image_path/question/answer/label)
라벨링
- --label_mode dataset : gold->1 / hallucinated->0
- --label_mode answer  : yes->1 / no->0

저장
- {out_dir}/raw.pt            -> {"raw_head_acts": (N,L,H,D), "labels": (N,)}
- {out_dir}/head_metrics.csv  -> per (layer, head) metrics
"""

import os, json, re, argparse
from typing import List, Dict, Tuple, Optional, Iterable
from io import BytesIO
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoProcessor, AutoModelForVision2Seq,
    AutoConfig, AutoModel
)

# Optional imports for Qwen2.5-Omni (if available)
try:
    from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLProcessor = None
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
except Exception:
    Qwen2_5OmniProcessor = None
    Qwen2_5OmniThinkerForConditionalGeneration = None

# ---------------------------
# Utils
# ---------------------------
def set_threads():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    print(f"[INFO] Loaded {len(rows)} rows from {path}")
    return rows

def build_prompt(context: str, question: str, answer: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}"

def label_from_dataset(row: Dict) -> int:
    val = str(row.get("label", "")).strip().lower()
    if val == "gold": return 1
    if val == "hallucinated": return 0
    raise ValueError(f"Unknown dataset label: {val}")

def label_from_answer(row: Dict) -> int:
    ans = str(row.get("answer", "")).strip().lower()
    if ans == "yes": return 1
    if ans == "no":  return 0
    raise ValueError(f"Answer must be yes/no for label_mode=answer, got: {ans}")

def make_black_image(w=336, h=336):
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")

def load_image_or_none(p: Optional[str]) -> Optional[Image.Image]:
    if not p or not os.path.exists(p):
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

# ---------------------------
# Hook (LLaMA/Vicuna-like)
# ---------------------------
class OProjInputCollector:
    """Collect input to self_attn.o_proj for each layer at a chosen token index."""
    def __init__(self, layers_container: nn.ModuleList, num_layers: int):
        self.num_layers = num_layers
        self.buffers = [None] * num_layers
        self.handles = []
        assert len(layers_container) == num_layers
        for i, layer in enumerate(layers_container):
            lin: nn.Linear = layer.self_attn.o_proj
            h = lin.register_forward_pre_hook(self._make_hook(i))
            self.handles.append(h)
        self.token_index = 0  # set per-forward

    def _make_hook(self, idx: int):
        def hook(module, inputs):
            x = inputs[0]  # (B, T, hidden)
            ti = max(0, min(self.token_index, x.shape[1]-1))
            self.buffers[idx] = x[:, ti, :].detach().to("cpu")
        return hook

    def clear(self):
        for i in range(self.num_layers):
            self.buffers[i] = None

    def remove(self):
        for h in self.handles:
            h.remove()

# ---------------------------
# Dataset builders
# ---------------------------
def build_rows_and_labels_cqa(args) -> Tuple[List[Tuple[str,str,str]], List[int]]:
    rows = load_jsonl(args.jsonl_path)
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    items, labels = [], []
    for r in rows:
        try:
            if args.label_mode == "dataset":
                y = label_from_dataset(r)
            else:
                y = label_from_answer(r)
        except ValueError:
            if args.skip_bad: continue
            else: raise

        c = str(r.get("context", "")).strip()
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        items.append((c, q, a))
        labels.append(y)
    print(f"[INFO] Kept {len(items)} usable samples")
    return items, labels

# ---------------------------
# Token index helpers
# ---------------------------
def last_token_index_from_inputs(inputs: Dict[str, torch.Tensor]) -> int:
    if "attention_mask" in inputs:
        am = inputs["attention_mask"][0]
        nz = torch.nonzero(am, as_tuple=False)
        if nz.numel() > 0:
            return int(nz[-1])
    return inputs["input_ids"].shape[1] - 1

def maybe_strip_eos(tok, input_ids: torch.Tensor, idx: int, strip: bool) -> int:
    if not strip: return idx
    try:
        eos_id = tok.eos_token_id
        if eos_id is None: return idx
        if idx > 0 and int(input_ids[0, idx].item()) == int(eos_id):
            return idx - 1
    except Exception:
        pass
    return idx

# ---------------------------
# Model loading helpers for Qwen2.5
# ---------------------------
def _infer_dtype(device, force_fp32=False):
    if device.type == "cuda" and not force_fp32:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def _detect_family(model_name: str) -> str:
    """Detect if model is qwen-vl or omni based on name/config."""
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        mt = getattr(cfg, "model_type", "") or ""
        if "omni" in mt.lower():
            return "omni"
        if "qwen2_5_vl" in mt.lower() or "qwen2_vl" in mt.lower() or "vision2seq" in mt.lower():
            return "qwen-vl"
    except Exception:
        pass
    nm = model_name.lower()
    if "omni" in nm:
        return "omni"
    if "vl" in nm:
        return "qwen-vl"
    return "qwen-vl"  # default

def get_qwen_vision_token_string(processor) -> str:
    """Qwen 계열이 기대하는 비전 토큰 시퀀스를 안전하게 구성 (원본 black.py와 동일)."""
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        add = (getattr(tok, "special_tokens_map", {}) or {}).get("additional_special_tokens", []) or []
        order = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
        found = [t for t in order if t in add]
        if found:
            return "".join(found)
    return "<|vision_start|><|image_pad|><|vision_end|>"

# ---------------------------
# LLM path
# ---------------------------
def collect_llm(args, device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Load Qwen2.5 text model
    tok = AutoTokenizer.from_pretrained(args.llm_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = _infer_dtype(device, args.force_fp32)

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=(dtype if device.type == "cuda" else torch.float32),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None
    ).eval()
    # If device_map was not used (CPU case), manually move to device
    if device.type != "cuda":
        model = model.to(device)

    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads
    hidden_size= model.config.hidden_size
    head_dim   = hidden_size // num_heads

    layers_container = model.model.layers  # LLaMA-family
    collector = OProjInputCollector(layers_container, num_layers)

    items, labels = build_rows_and_labels_cqa(args)

    head_acts = []
    with torch.no_grad():
        for c, q, a in tqdm(items, desc="Collect (LLM)"):
            text = build_prompt(c, q, a)
            enc = tok(text, return_tensors="pt", truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}

            last_idx = last_token_index_from_inputs(enc)
            last_idx = maybe_strip_eos(tok, enc["input_ids"], last_idx, args.strip_trailing_eos)

            collector.clear()
            collector.token_index = last_idx
            _ = model(**enc)

            per_layer = []
            for l in range(num_layers):
                vec = collector.buffers[l]
                if vec is None:
                    per_layer.append(torch.zeros((num_heads, head_dim)))
                else:
                    v = vec.squeeze(0).float().view(num_heads, head_dim)
                    per_layer.append(v)
            head_acts.append(torch.stack(per_layer, dim=0))  # (L,H,D)

    X = torch.stack(head_acts, dim=0)  # (N,L,H,D)
    y = torch.tensor(labels, dtype=torch.long)
    ensure_dir(args.out_dir)
    torch.save({"raw_head_acts": X, "labels": y}, os.path.join(args.out_dir, "raw.pt"))
    print("[INFO] Saved raw ->", os.path.join(args.out_dir, "raw.pt"))
    return X, y

# ---------------------------
# VLM commons
# ---------------------------
ATTN_ATTR_NAMES = ["self_attn","self_attention","attn","attention","mha"]
PROJ_ATTR_NAMES = ["o_proj","out_proj"]

def _get_attr_chain(obj, chain: str):
    cur = obj
    for part in chain.split("."):
        if not hasattr(cur, part): return None
        cur = getattr(cur, part)
    return cur

def _first_layers_container(lm) -> Optional[Iterable]:
    candidates = ["model.layers","layers","decoder.layers","transformer.layers","transformer.h"]
    for p in candidates:
        cont = _get_attr_chain(lm, p)
        if cont is not None and hasattr(cont, "__len__") and len(cont) > 0:
            return cont
    return None

def get_text_lm(model):
    lm = getattr(model, "language_model", None)
    if lm is None: lm = getattr(model, "model", None)
    return lm

def collect_text_o_proj_modules_generic(model) -> List[Tuple[str, nn.Module]]:
    lm = get_text_lm(model)
    if lm is None: raise RuntimeError("No language_model/model found in VLM.")
    layers = _first_layers_container(lm)
    candidates: List[Tuple[str, nn.Module]] = []
    if layers is not None:
        for i, blk in enumerate(layers):
            attn = None
            for name in ATTN_ATTR_NAMES:
                if hasattr(blk, name): attn = getattr(blk, name); break
            if attn is None:
                for n, m in blk.named_modules():
                    if any(k in n for k in ATTN_ATTR_NAMES): attn = m; break
            if attn is None: continue

            proj = None
            for name in PROJ_ATTR_NAMES:
                if hasattr(attn, name): proj = getattr(attn, name); break
            if proj is None:
                for n, m in attn.named_modules():
                    if ("proj" in n.lower()) and isinstance(m, nn.Linear):
                        proj = m; break
            if proj is not None:
                candidates.append((f"LM.layers[{i}].o_proj", proj))
    if not candidates:
        for n, m in lm.named_modules():
            ln = n.lower()
            if any(a in ln for a in ATTN_ATTR_NAMES) and any(p in ln for p in PROJ_ATTR_NAMES):
                if isinstance(m, nn.Linear):
                    candidates.append((f"LM.{n}", m))
    def layer_idx(name: str) -> int:
        m = re.search(r"layers?\[(\d+)\]", name)
        return int(m.group(1)) if m else 1_000_000
    candidates.sort(key=lambda x: layer_idx(x[0]))
    if not candidates: raise RuntimeError("No LM attn out-proj modules found.")
    return candidates

# ---------------------------
# VLM (C+Q+A 텍스트 기반: text_only / black)
# ---------------------------
def collect_vlm_text(args, device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Detect model family (qwen-vl or omni) -- collect_vlm_img_phd와 동일한 방식
    family = _detect_family(args.vlm_name)

    if family == "omni" and Qwen2_5OmniProcessor is not None:
        processor = Qwen2_5OmniProcessor.from_pretrained(args.vlm_name)
    else:
        try:
            processor = AutoProcessor.from_pretrained(args.vlm_name, use_fast=True, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(args.vlm_name, trust_remote_code=True)

    dtype = _infer_dtype(device, args.force_fp32)

    if family == "omni" and Qwen2_5OmniThinkerForConditionalGeneration is not None:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.vlm_name, torch_dtype=dtype, device_map="auto"
        ).eval()
    else:
        # Detect model family and load accordingly
        cfg = AutoConfig.from_pretrained(args.vlm_name, trust_remote_code=True)
        mt = (getattr(cfg, "model_type", "") or "").lower()

        if "qwen2_5_vl" in mt:
            model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()
        elif "qwen2_vl" in mt and Qwen2VLForConditionalGeneration is not None:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.vlm_name, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
            ).eval().to(device)
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
            ).eval()

    lm = get_text_lm(model)
    cfgs = []
    if lm is not None and hasattr(lm, "config"): cfgs.append(lm.config)
    if hasattr(model, "config") and hasattr(model.config, "text_config"): cfgs.append(model.config.text_config)
    if hasattr(model, "config"): cfgs.append(model.config)
    hidden_size = num_heads = None
    for c in cfgs:
        if c is None: continue
        if hidden_size is None and hasattr(c, "hidden_size"): hidden_size = c.hidden_size
        if num_heads  is None and hasattr(c, "num_attention_heads"): num_heads = c.num_attention_heads
    if hidden_size is None or num_heads is None:
        raise RuntimeError("Could not infer LM dims for VLM.")
    head_dim = hidden_size // num_heads

    candidates = collect_text_o_proj_modules_generic(model)
    hooks, hook_data, last_idx_box = [], {}, {"idx":0}

    def make_hook(name):
        def hook(module, inputs, output):
            t = None
            try: t = inputs[0]
            except Exception: t = output
            if t is None: return
            try:
                if t.dim()==3:
                    idx = min(last_idx_box["idx"], t.shape[1]-1)
                    slice_t = t[:, idx, :].detach()
                elif t.dim()==2:
                    idx = min(last_idx_box["idx"], t.shape[0]-1)
                    slice_t = t[idx, :].detach().unsqueeze(0)
                else:
                    return
                hook_data[name] = slice_t
            except Exception:
                pass
        return hook

    for name, module in candidates:
        hooks.append(module.register_forward_hook(make_hook(name)))

    # Build prompts from C+Q+A
    items, labels = build_rows_and_labels_cqa(args)
    prompts = []
    for c, q, a in items:
        text = build_prompt(c, q, a)
        if args.cond == "text_only":
            prompts.append((None, text))
        elif args.cond == "black":
            prompts.append((make_black_image(args.black_w, args.black_h), text))
        else:
            raise ValueError("Unsupported cond for VLM text mode (use text_only or black).")

    if args.max_samples>0 and len(prompts)>args.max_samples:
        prompts = prompts[:args.max_samples]; labels = labels[:args.max_samples]

    head_acts_list, failed = [], 0
    amp_ctx = torch.autocast("cuda", dtype=dtype) if (device.type=="cuda" and dtype!=torch.float32) else nullcontext()
    vis_token_str = get_qwen_vision_token_string(processor)
    print(f"[INFO] Collecting VLM (text) activations: N={len(prompts)} cond={args.cond}")
    for i, (img, text) in enumerate(tqdm(prompts, desc=f"VLM collect [{args.cond}]")):
        try:
            if img is not None:
                # 이미지가 있으면(black) Qwen이 기대하는 vision 특수토큰 문자열을 텍스트 앞에
                # 그대로 붙인다 (원본 black.py의 --prompt_mode plain과 동일). chat template은
                # text_only와 구조 자체가 달라져 probe 결과가 왜곡되므로 쓰지 않는다.
                inputs = processor(text=f"{vis_token_str}\n{text}", images=[img], return_tensors="pt")
            else:
                inputs = processor(text=text, images=None, return_tensors="pt")
        except Exception as e:
            print(f"[WARN] processor failed on sample {i}: {e}"); failed+=1; head_acts_list.append(None); continue
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if "attention_mask" in inputs:
            am = inputs["attention_mask"][0]
            nz = torch.nonzero(am, as_tuple=False)
            last_idx = int(nz[-1]) if nz.numel()>0 else inputs["input_ids"].shape[1]-1
        else:
            last_idx = inputs["input_ids"].shape[1]-1
        last_idx_box["idx"] = last_idx

        hook_data.clear()
        try:
            with torch.inference_mode(), amp_ctx:
                _ = model(**inputs, use_cache=False)
        except Exception as e:
            print(f"[WARN] model call failed on sample {i}: {e}")
            failed+=1; head_acts_list.append(None); continue

        per_layer = []
        for name, _ in candidates:
            t = hook_data.get(name, None)
            if t is None: per_layer.append(None); continue
            vec = t[0]
            if vec.shape[0] != hidden_size:
                raise RuntimeError(f"Unexpected vec dim at {name}: {vec.shape[0]} vs hidden {hidden_size}")
            per_layer.append(vec.view(num_heads, head_dim))
        Hc=Dc=None
        for v in per_layer:
            if isinstance(v, torch.Tensor) and v.ndim==2: Hc,Dc=v.shape; break
        if Hc is None: head_acts_list.append(None); failed+=1; continue
        Lc = len(per_layer)
        arr = torch.zeros((Lc, Hc, Dc), dtype=torch.float32, device=device)
        for li, v in enumerate(per_layer):
            if v is not None: arr[li] = v.to(dtype=arr.dtype)
        head_acts_list.append(arr)

    print(f"[INFO] VLM(text) collection done. failed={failed}/{len(prompts)}")
    for h in hooks:
        try: h.remove()
        except: pass

    good = [i for i,a in enumerate(head_acts_list) if a is not None]
    if not good: raise RuntimeError("No valid activations collected for VLM(text).")
    X = torch.stack([head_acts_list[i] for i in good], dim=0)   # (N,L,H,D)
    y = torch.tensor([labels[i] for i in good], device=device, dtype=torch.long)

    ensure_dir(args.out_dir)
    torch.save({"raw_head_acts": X, "labels": y}, os.path.join(args.out_dir, "raw.pt"))
    print("[INFO] Saved raw ->", os.path.join(args.out_dir, "raw.pt"))
    return X, y

# ---------------------------
# VLM (이미지 PHD 스타일: --img)
# ---------------------------
def collect_vlm_img_phd(args, device):
    # Detect model family (qwen-vl or omni)
    family = _detect_family(args.vlm_name)
    
    # Load processor
    if family == "omni" and Qwen2_5OmniProcessor is not None:
        processor = Qwen2_5OmniProcessor.from_pretrained(args.vlm_name)
    else:
        try:
            processor = AutoProcessor.from_pretrained(args.vlm_name, use_fast=True, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(args.vlm_name, trust_remote_code=True)

    dtype = _infer_dtype(device, args.force_fp32)
    
    # Load model based on family
    if family == "omni" and Qwen2_5OmniThinkerForConditionalGeneration is not None:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.vlm_name, torch_dtype=dtype, device_map="auto"
        ).eval()
    else:
        # Qwen2.5-VL or fallback
        cfg = AutoConfig.from_pretrained(args.vlm_name, trust_remote_code=True)
        mt = (getattr(cfg, "model_type", "") or "").lower()
        
        if "qwen2_5_vl" in mt:
            model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()
        elif "qwen2_vl" in mt and Qwen2VLForConditionalGeneration is not None:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.vlm_name, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
            ).eval().to(device)
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                args.vlm_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
            ).eval()

    lm = get_text_lm(model)
    layers = _first_layers_container(lm)
    if layers is None:
        raise RuntimeError("LM layers not found in VLM.")
    hidden_size = getattr(lm.config, "hidden_size", None) or getattr(model.config, "hidden_size", None)
    num_heads  = getattr(lm.config, "num_attention_heads", None) or getattr(model.config, "num_attention_heads", None)
    if hidden_size is None or num_heads is None:
        raise RuntimeError("Could not infer hidden_size/num_heads.")
    head_dim = hidden_size // num_heads

    # register hooks
    collector = OProjInputCollector(layers ,len(layers))

    rows = load_jsonl(args.jsonl_path)
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    head_acts, labels, skipped = [], [], 0

    with torch.no_grad():
        for r in tqdm(rows, desc="Collect (VLM: IMG-PHD)"):
            img = load_image_or_none(r.get("image_path"))
            if img is None:
                skipped += 1
                continue

            question = str(r.get("question", "")).strip()
            answer   = str(r.get("answer", "")).strip()
            
            # Use chat template format for Qwen2.5-VL (properly handles image tokens)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
            
            try:
                # Try chat template first (recommended for Qwen2.5-VL)
                tok = getattr(processor, "tokenizer", None)
                if tok is not None and hasattr(tok, "apply_chat_template"):
                    try:
                        text = tok.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        enc = processor(text=text, images=[img], return_tensors="pt")
                    except Exception as e:
                        # Fallback to plain format
                        prompt = f"<image>\n{question}\nASSISTANT: {answer}"
                        enc = processor(text=prompt, images=[img], return_tensors="pt")
                else:
                    # Fallback to plain format if chat template not available
                    prompt = f"<image>\n{question}\nASSISTANT: {answer}"
                    enc = processor(text=prompt, images=[img], return_tensors="pt")
            except Exception as e:
                print(f"[WARN] Processing failed for sample: {e}")
                skipped += 1
                continue

            enc = {k: v.to(device) for k, v in enc.items()}
            last_idx = last_token_index_from_inputs(enc)
            collector.clear()
            collector.token_index = last_idx

            _ = model(**enc, use_cache=False)

            per_layer = []
            for l in range(collector.num_layers):
                vec = collector.buffers[l]
                if vec is None:
                    per_layer.append(torch.zeros((num_heads, head_dim)))
                else:
                    v = vec.squeeze(0).float().view(num_heads, head_dim)
                    per_layer.append(v)
            head_acts.append(torch.stack(per_layer, dim=0))

            # 라벨
            try:
                y = label_from_dataset(r) if args.label_mode=="dataset" else label_from_answer(r)
            except ValueError:
                if args.skip_bad:
                    head_acts.pop()
                    skipped += 1
                    continue
                else:
                    raise
            labels.append(y)

    if not head_acts:
        raise RuntimeError("No valid samples collected (all skipped?).")

    X = torch.stack(head_acts, dim=0)  # (N,L,H,D)
    y = torch.tensor(labels, dtype=torch.long)

    ensure_dir(args.out_dir)
    torch.save({"raw_head_acts": X, "labels": y}, os.path.join(args.out_dir, "raw.pt"))
    print(f"[INFO] Saved raw -> {os.path.join(args.out_dir, 'raw.pt')}")
    print(f"[INFO] Skipped {skipped} samples due to missing/failed images.")
    return X, y

# ---------------------------
# Linear probe (vectorized)
# ---------------------------
def vectorized_linear_probe(X: torch.Tensor, y: torch.Tensor, args, device) -> pd.DataFrame:
    X = X.to(device, dtype=torch.float32)
    y = y.to(device)

    N, L, H, D = X.shape
    idx_all = list(range(N))
    
    # Set 5-fold cross validation 
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    # Initialize dictionary to store metrics for each fold
    fold_metrics = {}
    
    # Iterate over 5 folds
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(idx_all)):
        print(f"Fold {fold_idx + 1}/5")

        # tr_choice = [i for q in train_idx for i in idx_all[q]]
        # va_choice = [i for q in val_idx for i in idx_all[q]]
        tr_choice = list(train_idx)
        va_choice = list(val_idx)

        N, L, H, D = X.shape
        W = torch.zeros(L, H, D, device=device, requires_grad=True)
        B = torch.zeros(L, H, device=device, requires_grad=True)
        opt = torch.optim.AdamW([W, B], lr=args.probe_lr, weight_decay=args.probe_weight_decay)

        def iterb(idxs, bs):
            for s in range(0, len(idxs), bs):
                yield idxs[s:s+bs]

        # train
        for ep in range(args.probe_epochs):
            runloss = 0.0
            for batch in iterb(tr_choice, args.probe_batch_size):
                Xb = X[batch]           # (B,L,H,D)
                yb = y[batch].float()   # (B,)
                logits = (Xb * W).sum(-1) + B    # (B,L,H)
                target = yb[:, None, None].expand(-1, L, H)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                runloss += float(loss.detach())
            if (ep + 1) % max(1, args.probe_epochs // 5) == 0:
                print(f"[PROBE] {ep+1}/{args.probe_epochs} loss={runloss:.4f}")

        # eval
        preds, trues = [], []
        with torch.inference_mode():
            for batch in iterb(va_choice, args.probe_batch_size):
                Xb = X[batch]
                yb = y[batch]
                prob = torch.sigmoid((Xb * W).sum(-1) + B).to("cpu").numpy()  # (B,L,H)
                pred = (prob > 0.5).astype(np.int32)
                preds.append(pred)
                trues.append(yb.to("cpu").numpy())
        y_va = np.concatenate(trues, 0)     # (B,)
        yhat = np.concatenate(preds, 0)     # (B,L,H)

        for l in range(L):
            for h in range(H):
                if (l, h) not in fold_metrics.keys():
                    fold_metrics[(l, h)] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
                yp = yhat[:, l, h]
                acc = float(accuracy_score(y_va, yp))
                pre = float(precision_score(y_va, yp, zero_division=0))
                rec = float(recall_score(y_va, yp, zero_division=0))
                f1  = float(f1_score(y_va, yp, zero_division=0))
                
                fold_metrics[(l, h)]["accuracy"].append(acc)
                fold_metrics[(l, h)]["precision"].append(pre)
                fold_metrics[(l, h)]["recall"].append(rec)
                fold_metrics[(l, h)]["f1"].append(f1)
    # Calculate mean metrics for each fold
    for l in range(L):
        for h in range(H):
            fold_metrics[(l, h)]["accuracy"] = np.mean(fold_metrics[(l, h)]["accuracy"])
            fold_metrics[(l, h)]["precision"] = np.mean(fold_metrics[(l, h)]["precision"])
            fold_metrics[(l, h)]["recall"] = np.mean(fold_metrics[(l, h)]["recall"])
            fold_metrics[(l, h)]["f1"] = np.mean(fold_metrics[(l, h)]["f1"])

    # Convert dictionary to DataFrame
    rows_m = []
    for l in range(L):
        for h in range(H):
            rows_m.append({"layer": l, "head": h,
                           "accuracy": fold_metrics[(l, h)]["accuracy"], "precision": fold_metrics[(l, h)]["precision"], "recall": fold_metrics[(l, h)]["recall"], "f1": fold_metrics[(l, h)]["f1"]})

    df = pd.DataFrame(rows_m)
    df.to_csv(os.path.join(args.out_dir, f"cv5_head_metrics.csv"), index=False)
    print("[INFO] Saved metrics ->", os.path.join(args.out_dir, f"cv5_head_metrics.csv"))
    return df

# ---------------------------
# Main
# ---------------------------
def main():
    set_threads()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm","vlm"], default="llm")
    ap.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B")
    ap.add_argument("--vlm_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--jsonl_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./out_probe_unified")

    # VLM 전용
    ap.add_argument("--cond", choices=["text_only","black"], default="text_only",
                    help="VLM 텍스트 경로 전용. --img가 켜져있으면 무시")
    ap.add_argument("--black_w", type=int, default=336)
    ap.add_argument("--black_h", type=int, default=336)
    ap.add_argument("--img", action="store_true",
                    help="켜면 이미지(PHD 포맷: image_path/question/answer/label) 모드로 수집")

    # labeling
    ap.add_argument("--label_mode", choices=["dataset", "answer"], default="dataset",
                    help="dataset: gold/hallucinated -> 1/0, answer: yes/no -> 1/0")
    ap.add_argument("--skip_bad", action="store_true", help="Skip rows with invalid/missing labels")
    ap.add_argument("--max_samples", type=int, default=0)

    # probe hyperparams
    ap.add_argument("--probe_epochs", type=int, default=200)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_weight_decay", type=float, default=1e-3)
    ap.add_argument("--probe_batch_size", type=int, default=512)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--force_fp32", action="store_true", help="Force float32 on CPU/GPU")
    ap.add_argument("--strip_trailing_eos", action="store_true",
                    help="LLM 전용: 마지막 토큰이 EOS면 이전 토큰으로 이동")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    ensure_dir(args.out_dir)

    if args.mode == "llm":
        X, y = collect_llm(args, device)
    else:
        if args.img:
            X, y = collect_vlm_img_phd(args, device)
        else:
            X, y = collect_vlm_text(args, device)

    _ = vectorized_linear_probe(X, y, args, device)
    print("✅ Done.")

if __name__ == "__main__":
    main()
