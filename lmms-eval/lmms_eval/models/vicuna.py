import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM 

# Import our custom model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "LLaVA-NeXT", "llava", "model", "language_model"))
from vicuna_llama import VicunaCustomForCausalLM

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria


warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


@register_model("vicuna")
class vicuna(lmms):
    """
    Vicuna Model "https://github.com/lm-sys/FastChat"
    """

    def __init__(
        self,
        pretrained: str = "lmsys/vicuna-7b-v1.5",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = "eager",
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "vicuna_v1",
        use_cache: Optional[bool] = True,
        cfg=None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Store the config
        self.cfg = cfg or {}

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Load the model with custom wrapper
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained)
        config.model_type = "vicuna_llama"
        self._model = VicunaCustomForCausalLM.from_pretrained(
            pretrained,
            config=config,
            device_map=self._device,
            attn_implementation=attn_implementation
        )

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, attn_implementation=attn_implementation)
        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
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
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if isinstance(doc_to_target, str):
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            try:
                continuation = int(continuation)
            except (ValueError, TypeError):
                pass

            prompts_text = contexts[0] if isinstance(contexts, list) else contexts

            full_input = prompts_text + str(continuation)

            input_ids = self.tokenizer(full_input, return_tensors="pt").input_ids.to(self.device)

            context_len = len(self.tokenizer(prompts_text)["input_ids"])

            labels = input_ids.clone()
            labels[:, :context_len] = -100  

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, use_cache=self.use_cache)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]

            pred_tokens = logits.argmax(dim=-1)
            target_tokens = input_ids[:, context_len:]
            pred_cont = pred_tokens[:, context_len:]
 
            is_greedy = (pred_cont == target_tokens).all()
            res.append((loss.item(), bool(is_greedy)))
            pbar.update(1)

        pbar.close()
        return res


    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance], mask_info: dict = None) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs,doc_id,doc_to_visual,task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = all_gen_kwargs[0]
        
            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            if isinstance(contexts, tuple):
                contexts = list(contexts)
            context = contexts[0]

            # Tokenize prompt
            inputs = self._tokenizer(context, return_tensors="pt", padding=True, truncation=True).to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            try:
                if self.hyperparams.get("adaptive_max_new_tokens", False):
                    max_new_tokens = inputs['input_ids'].shape[-1] + 50
                else:
                    max_new_tokens = gen_kwargs["max_new_tokens"]
                cont = self.model.generate(
                    **inputs,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=max_new_tokens,
                    use_cache=self.use_cache,
                    gate_truthful_head=self.gate_truthful_head,
                    truthful_head=self.truthful_head,
                    hyperparams=self.hyperparams,
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                print(f"Generated answer: {ans}")
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        
        return res


    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for InstructBlip")

    @classmethod
    def from_config(cls, cfg, model_args=None):
        """
        Instantiate from config dictionary.
        
        Args:
            cfg: Task configuration dictionary
            model_args: Model arguments string (e.g., "pretrained=lmsys/vicuna-7b-v1.5")
        """

        # Parse model arguments if provided
        if model_args:
            from lmms_eval.utils import simple_parse_args_string
            parsed_model_args = simple_parse_args_string(model_args)
            pretrained = parsed_model_args.get("pretrained", "lmsys/vicuna-7b-v1.5")
            device = parsed_model_args.get("device", "cuda:0")
            batch_size = parsed_model_args.get("batch_size", 1)
            use_cache = parsed_model_args.get("use_cache", True)
        else:
            pretrained = "lmsys/vicuna-7b-v1.5"
            device = "cuda:0"
            batch_size = 1
            use_cache = True
        
        return cls(
            pretrained=pretrained,  # Extract from model_args instead of hardcoding
            device=device,
            batch_size=batch_size,
            use_cache=use_cache,
            cfg=cfg,  # Pass the full task config
        )
