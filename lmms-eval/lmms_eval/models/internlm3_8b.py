import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "LLaVA-NeXT", "llava", "model", "language_model"))
from internlm3_llama import InternLM3CustomForCausalLM


@register_model("internlm3_8b")
class InternLM3_8B(lmms):
    """
    InternLM3-8B-Instruct text model wrapper for LMMS-Eval.
    """

    def __init__(
        self,
        pretrained: str = "internlm/internlm3-8b-instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        system_prompt: Optional[str] = (
            "You are InternLM, a helpful, honest, and harmless AI assistant. \n"
            "Always respond in the same language the user is writing in."
        ),
        cfg=None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.cfg = cfg or {}
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {"torch_dtype": "auto", "device_map": self.device_map}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        config.model_type = "internlm3_llama"
        self._model = InternLM3CustomForCausalLM.from_pretrained(pretrained, config=config, **model_kwargs).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.system_prompt = system_prompt
        self._config = getattr(self.model, "config", config)
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], (
                "Unsupported distributed type provided. Only DDP and FSDP are supported."
            )
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        metadata = self.cfg.get("metadata", {}) or {}
        self.cfg["metadata"] = metadata
        self.gate_truthful_head = metadata.get("gate_truthful_head", False)
        truthful_head_filepath = metadata.get("truthful_head_filepath", None)
        if truthful_head_filepath is not None:
            import numpy as np

            self.truthful_head = torch.from_numpy(np.load(truthful_head_filepath))
        else:
            self.truthful_head = None

        self.hyperparams = metadata.get("hyperparams", {}) or {}

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, _doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            continuation = doc_to_target if isinstance(doc_to_target, str) else doc_to_target(self.task_dict[task][split][doc_id])
            prompts_text = contexts[0] if isinstance(contexts, list) else contexts
            full_input = prompts_text + str(continuation)
            input_ids = self.tokenizer(full_input, return_tensors="pt").input_ids.to(self.device)
            context_len = len(self.tokenizer(prompts_text)["input_ids"])

            labels = input_ids.clone()
            labels[:, :context_len] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels)

            loss = outputs["loss"]
            logits = outputs["logits"]
            pred_tokens = logits.argmax(dim=-1)
            target_tokens = input_ids[:, context_len:]
            greedy_tokens = pred_tokens[:, context_len : input_ids.shape[1]]
            max_equal = (greedy_tokens == target_tokens).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(self.tokenizer.encode(x[0])), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, _doc_to_visual, _doc_id, _task, _split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], but got {type(until)}")

            contexts = list(contexts) if isinstance(contexts, tuple) else contexts
            texts = [str(ctx).strip() for ctx in contexts]

            # Do not apply chat template; prepend system prompt explicitly to avoid template control tokens.
            prompt_texts = [f"{self.system_prompt}\n\n{text}" for text in texts]
            inputs = self.tokenizer(text=prompt_texts, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)

            # Set config in official implementation
            # (tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)
            default_gen_kwargs = {
                "max_new_tokens": 6,
                "temperature": 1.0,
                "top_p": 0.8,
                "top_k": 40,
                "repetition_penalty": 1.005,
                "num_beams": 1,
            }
            # current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            current_gen_kwargs = default_gen_kwargs
            pad_token_id = self.tokenizer.pad_token_id

            do_sample = current_gen_kwargs["temperature"] > 0
            if not do_sample:
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=do_sample,
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                gate_truthful_head=self.gate_truthful_head,
                truthful_head=self.truthful_head,
                hyperparams=self.hyperparams,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    @classmethod
    def from_config(cls, cfg, model_args=None):
        if model_args:
            from lmms_eval.utils import simple_parse_args_string

            parsed_model_args = simple_parse_args_string(model_args)
            pretrained = parsed_model_args.get("pretrained", "internlm/internlm3-8b-instruct")
            device = parsed_model_args.get("device", "cuda")
            device_map = parsed_model_args.get("device_map", "auto")
            batch_size = parsed_model_args.get("batch_size", 1)
            attn_implementation = parsed_model_args.get("attn_implementation", None)
            use_cache = parsed_model_args.get("use_cache", True)
            system_prompt = parsed_model_args.get(
                "You are InternLM, a helpful, honest, and harmless AI assistant. \n"
                "Always respond in the same language the user is writing in.",
            )
        else:
            pretrained = "internlm/internlm3-8b-instruct"
            device = "cuda"
            device_map = "auto"
            batch_size = 1
            attn_implementation = None
            use_cache = True
            system_prompt = (
                "You are InternLM, a helpful, honest, and harmless AI assistant. \n"
                "Always respond in the same language the user is writing in.",
            )

        return cls(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            system_prompt=system_prompt,
            cfg=cfg,
        )
