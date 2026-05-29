import base64
import os
import re
import sys
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "LLaVA-NeXT", "llava", "model", "language_model"))
from qwen3_vl_instruct_llama import Qwen3VLInstructCustomForCausalLM

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_instruct")
class Qwen3_VL_Instruct(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        cfg=None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        self.cfg = cfg or {}

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained)
        config.model_type = "qwen3_vl_instruct_llama"
        self._model = Qwen3VLInstructCustomForCausalLM.from_pretrained(
            pretrained,
            config=config,
            **model_kwargs,
        ).eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self._config = self.model.config
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_VL_Instruct")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)
            contexts = [c.replace("<image>", "") if "<image>" in c else c for c in contexts]

            batched_messages = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": ''}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        vr = decord.VideoReader(visual)
                        _ = vr[0].asnumpy()
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )

                if self.interleave_visuals is False:
                    message.append({"role": "user", "content": processed_visuals + [{"type": "text", "text": context}]})
                else:
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})
                    for j, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if j + 1 < len(text_parts) and text_parts[j + 1]:
                            content_parts.append({"type": "text", "text": text_parts[j + 1]})
                    message.append({"role": "user", "content": content_parts})

                batched_messages.append(message)

            texts = [
				self.processor.apply_chat_template(
									msg, 
									tokenize=False,
									add_generation_prompt=True,
									enable_thinking=False) 
									# Switches between thinking and non-thinking modes. Default is True.
					for msg in batched_messages
				]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]

            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)

			# Set default generation kwargs (for non-thinking mode)
            default_gen_kwargs = {
                "max_new_tokens": 64,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "num_beams": 1,
            }
            # current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            current_gen_kwargs = default_gen_kwargs
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
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
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
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
            pretrained = parsed_model_args.get("pretrained", "Qwen/Qwen3-VL-8B-Instruct")
            device = parsed_model_args.get("device", "cuda")
            device_map = parsed_model_args.get("device_map", "auto")
            batch_size = parsed_model_args.get("batch_size", 1)
            attn_implementation = parsed_model_args.get("attn_implementation", None)
            use_cache = parsed_model_args.get("use_cache", True)
            min_pixels = parsed_model_args.get("min_pixels", 256 * 28 * 28)
            max_pixels = parsed_model_args.get("max_pixels", 1605632)
            max_num_frames = parsed_model_args.get("max_num_frames", 32)
            use_custom_video_loader = parsed_model_args.get("use_custom_video_loader", False)
            fps = parsed_model_args.get("fps", None)
            max_image_size = parsed_model_args.get("max_image_size", None)
            system_prompt = parsed_model_args.get("system_prompt", "You are a helpful assistant.")
            interleave_visuals = parsed_model_args.get("interleave_visuals", False)
            reasoning_prompt = parsed_model_args.get("reasoning_prompt", None)
        else:
            pretrained = "Qwen/Qwen3-VL-8B-Instruct"
            device = "cuda"
            device_map = "auto"
            batch_size = 1
            attn_implementation = None
            use_cache = True
            min_pixels = 256 * 28 * 28
            max_pixels = 1605632
            max_num_frames = 32
            use_custom_video_loader = False
            fps = None
            max_image_size = None
            system_prompt = "You are a helpful assistant."
            interleave_visuals = False
            reasoning_prompt = None

        return cls(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            use_custom_video_loader=use_custom_video_loader,
            fps=fps,
            max_image_size=max_image_size,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            cfg=cfg,
        )
