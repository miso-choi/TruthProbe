import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

# Import our custom model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../LLaVA-NeXT/llava/model/language_model'))
from qwen2_5_vl_llama import Qwen2_5_VLCustomForCausalLM

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _pope_image_source_to_coco_rel_path(image_source: str) -> str:
    """POPE `image_source` is often a stem like COCO_val2014_000000310196 (no folder/ext)."""
    s = (image_source or "").replace("\\", "/").strip()
    if not s:
        return ""
    low = s.lower()
    if "/" in s and low.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return s
    base = s.split("/")[-1]
    if base.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        if "/" in s:
            return s
        if base.startswith("COCO_val2014"):
            return f"val2014/{base}"
        if base.startswith("COCO_train2014"):
            return f"train2014/{base}"
        if base.startswith("COCO_test2014"):
            return f"test2014/{base}"
        return base
    if base.startswith("COCO_val2014"):
        return f"val2014/{base}.jpg"
    if base.startswith("COCO_train2014"):
        return f"train2014/{base}.jpg"
    if base.startswith("COCO_test2014"):
        return f"test2014/{base}.jpg"
    return f"{base}.jpg" if "." not in base else base


def _dataset_relative_image_path_for_jsonl(doc: dict) -> Tuple[str, str]:
    """
    Returns (rel_path_under_images/, raw_image_source_or_empty).
    CHAIR uses image_path; POPE uses image_source stem + COCO layout.
    """
    if doc.get("image_path"):
        return str(doc["image_path"]).strip(), (doc.get("image_source") or "")
    raw_src = (doc.get("image_source") or "").strip()
    if raw_src:
        return _pope_image_source_to_coco_rel_path(raw_src), raw_src
    return "", ""


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        cfg=None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.cfg = cfg or {}

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
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

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self._model = Qwen2_5_VLCustomForCausalLM.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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

        self.model_name = pretrained.replace("/", "__")
        self._attn_impl = attn_implementation
        
        self.hyperparams = metadata.get("hyperparams", {}) or {}

    def _qwen25_vl_visual_token_bounds(
        self, input_ids_1d: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> Tuple[int, int, int]:
        """
        Return [start, end) token indices in `input_ids` for the first image's visual tokens.
        """
        model = self.model
        cfg = model.config
        merge = int(model.visual.spatial_merge_size)
        n_vis = int(image_grid_thw[0].prod().item() // (merge**2))
        ids = input_ids_1d
        vs = cfg.vision_start_token_id
        pos_vs = (ids == vs).nonzero(as_tuple=True)[0]
        if len(pos_vs) == 0:
            raise ValueError("vision_start_token_id not found in input_ids")
        start = int(pos_vs[0].item()) + 1
        end = start + n_vis
        return start, end, n_vis

    @staticmethod
    def _attn_prefill_last_query_to_visual_range(
        layer_attns: Tuple[torch.Tensor, ...], batch_idx: int, vis_start: int, vis_end: int
    ) -> torch.Tensor:
        """
        Prefill step: each layer tensor (B, H, Q, K). Take last query row, keys over visual range.
        Returns float tensor (num_layers, H, n_vis).
        """
        out: List[torch.Tensor] = []
        for attn in layer_attns:
            # (B, H, Q, K)
            q_len = attn.shape[2]
            q_idx = -1 if q_len > 1 else 0
            sl = attn[batch_idx, :, q_idx, vis_start:vis_end].detach().float().cpu()
            out.append(sl)
        return torch.stack(out, dim=0)


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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        vr = decord.VideoReader(visual)
                        first_frame = vr[0].asnumpy()
                        height, width = first_frame.shape[:2]
                        # max_pixels = height * width
                        processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                    elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
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
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def _single_request_to_processor_inputs(self, request: Instance):
        """Build processor tensors for one generate_until request (batch size 1)."""
        context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
        doc = self.task_dict[task][split][doc_id]
        visual_list = doc_to_visual(doc)
        contexts = [context]
        if isinstance(contexts[0], str) and "<image>" in contexts[0]:
            contexts[0] = contexts[0].replace("<image>", "")

        batched_messages = []
        for i, ctx in enumerate(contexts):
            ctx = ctx.replace("<image>", "") if "<image>" in ctx else ctx
            message = [{"role": "system", "content": self.system_prompt}]
            if self.reasoning_prompt:
                ctx = ctx.strip() + self.reasoning_prompt
                contexts[i] = ctx

            processed_visuals = []
            for visual in visual_list:
                if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
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
                    base64_bytes = base64.b64encode(buffer.getvalue())
                    base64_string = base64_bytes.decode("utf-8")
                    processed_visuals.append(
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{base64_string}",
                            "max_pixels": self.max_pixels,
                            "min_pixels": self.min_pixels,
                        }
                    )

            if self.interleave_visuals is False:
                message.append(
                    {"role": "user", "content": processed_visuals + [{"type": "text", "text": ctx}]}
                )
            else:
                image_placeholders = re.findall(r"<image \d+>", ctx)
                content_parts = []
                text_parts = re.split(r"<image \d+>", ctx)
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

        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
        image_inputs, video_inputs = process_vision_info(batched_messages)
        if video_inputs is not None:
            total_frames = video_inputs[0].shape[0]
            indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
            if total_frames - 1 not in indices:
                indices = np.append(indices, total_frames - 1)
            video_inputs[0] = video_inputs[0][indices]
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)
        return inputs, contexts[0], gen_kwargs

    def generate_until_with_vqa_save_attention(self, requests: List[Instance]) -> List[str]:
        """
        Save attention from the prefill forward (last query position) to visual tokens only,
        matching the first-token prediction. Writes one .pt per sample plus JSONL metadata with image_path.
        Requires attn_implementation=eager and batch_size 1 recommended.
        """
        if self._attn_impl not in (None, "eager"):
            eval_logger.warning(
                "Attention weights need eager attention; got attn_implementation=%s. "
                "Use model_args pretrained=...,attn_implementation=eager",
                self._attn_impl,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta = self.cfg.get("metadata", {}) or {}
        base_output_dir = meta.get("output_dir") or os.getcwd()
        output_dir = os.path.join(base_output_dir, f"vqa_save_attention_results/{self.model_name}")
        os.makedirs(output_dir, exist_ok=True)
        task_name = "unknown_task"
        if len(requests) > 0:
            try:
                task_name = str(requests[0].args[4])
            except Exception:
                pass
        safe_task_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", task_name)
        # Per-task subdir so e.g. pope_save_attention and pope_generic_caption_save_attention
        # do not overwrite the same doc_*.pt under samples/attn/
        attn_samples_dir = os.path.join(output_dir, "samples", safe_task_name, "attn")
        os.makedirs(attn_samples_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{safe_task_name}_first_token_visual_attn_{timestamp}.jsonl")
        eval_logger.info("Saving first-token visual attention to %s", output_dir)

        for request in tqdm(requests, total=len(requests), disable=(self.rank != 0), desc="Save attention"):
            _context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            image_path, _ = _dataset_relative_image_path_for_jsonl(doc)
            image_id = doc.get("image_id", None)
            question_id = doc.get("question_id", None)
            question = doc.get("question", "")

            try:
                inputs, _, gk = self._single_request_to_processor_inputs(request)
                image_grid_thw = getattr(inputs, "image_grid_thw", None)
                if image_grid_thw is None:
                    raise ValueError("image_grid_thw missing (image required for visual attention)")

                default_gen_kwargs = {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs, **gk}
                pad_token_id = self.tokenizer.pad_token_id
                if current_gen_kwargs["temperature"] and current_gen_kwargs["temperature"] > 0:
                    current_gen_kwargs["do_sample"] = True
                else:
                    current_gen_kwargs["do_sample"] = False
                    current_gen_kwargs["temperature"] = None
                    current_gen_kwargs["top_p"] = None

                vis_start, vis_end, n_vis = self._qwen25_vl_visual_token_bounds(inputs.input_ids[0], image_grid_thw)

                out = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    gate_truthful_head=self.gate_truthful_head,
                    truthful_head=self.truthful_head,
                    hyperparams=self.hyperparams,
                )

                if not getattr(out, "attentions", None) or len(out.attentions) == 0:
                    raise RuntimeError(
                        "generate() returned no attentions. Load the model with attn_implementation=eager."
                    )
                step0 = out.attentions[0]
                attn_to_visual = self._attn_prefill_last_query_to_visual_range(step0, 0, vis_start, vis_end)

                pt_name = f"doc_{doc_id}_first_token_vis_attn.pt"
                pt_path = os.path.join(attn_samples_dir, pt_name)
                payload = {
                    "attn_to_visual": attn_to_visual,
                    "vis_token_start": vis_start,
                    "vis_token_end": vis_end,
                    "num_visual_tokens": n_vis,
                    "num_layers": attn_to_visual.shape[0],
                    "num_heads": attn_to_visual.shape[1],
                    "note": "Prefill last-query position attention to visual tokens (first token prediction); float32 CPU",
                }
                torch.save(payload, pt_path)

                row = {
                    "doc_id": doc_id,
                    "task": task,
                    "split": split,
                    "image_path": image_path,
                    "image_source": (doc.get("image_source") or ""),
                    "image_id": image_id,
                    "question_id": question_id,
                    "question": question,
                    "attention_pt": pt_path,
                    "attention_shape_layers_heads_vis": list(attn_to_visual.shape),
                    "timestamp": timestamp,
                }
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                del out
            except RuntimeError as e:
                eval_logger.error("save_attention failed doc_id=%s: %s", doc_id, e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return [json.dumps({"status": "completed", "jsonl": output_file, "num_samples": len(requests)})]

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    @classmethod
    def from_config(cls, cfg, model_args=None):
        """
        Instantiate from config dictionary.
        
        Args:
            cfg: Task configuration dictionary
            model_args: Model arguments string (e.g., "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=eager")
        """
        # Parse model arguments if provided
        if model_args:
            from lmms_eval.utils import simple_parse_args_string
            parsed_model_args = simple_parse_args_string(model_args)
            pretrained = parsed_model_args.get("pretrained", "Qwen/Qwen2.5-VL-7B-Instruct")
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
            pretrained = "Qwen/Qwen2.5-VL-7B-Instruct"
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
            cfg=cfg,  # Pass the full task config
        )
