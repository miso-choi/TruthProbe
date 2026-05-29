import gc
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list


def split_model(model_name, num_layers=None):
    device_map = {}
    world_size = torch.cuda.device_count()
    if num_layers is None:
        num_layers = {
            "InternVL3-1B": 24,
            "InternVL3-2B": 24,
            "InternVL3-8B": 32,
            "InternVL3-9B": 36,
            "InternVL3-14B": 40,
            "InternVL3-38B": 64,
            "InternVL3-78B": 80,
        }.get(model_name, 36)
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1

    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


@register_model("internvl3_9b")
class InternVL3_9B(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3-9B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_frame: int = 32,
        num_layers=None,
        attn_implementation: Optional[str] = None,
        load_in_8bit: bool = False,
        image_max_num: int = 6,
        image_input_size: int = 448,
        clear_cuda_cache_each_step: bool = False,
        cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.path = pretrained
        self.modality = modality
        self.num_frame = num_frame
        self.cfg = cfg or {}
        self.image_max_num = int(image_max_num)
        self.image_input_size = int(image_input_size)
        self.clear_cuda_cache_each_step = bool(clear_cuda_cache_each_step)

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL3, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs()
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = split_model(pretrained.split("/")[-1], num_layers=num_layers)
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        model_kwargs = dict(
            pretrained_model_name_or_path=self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device_map,
        )
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self._model = AutoModel.from_pretrained(**model_kwargs).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=False)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED]
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **ds_kwargs)
                eval_logger.info("Using DEEPSPEED. Ensure accelerate config zero stage is set to 0.")

            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
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

    @classmethod
    def from_config(cls, cfg, model_args=None):
        if model_args:
            from lmms_eval.utils import simple_parse_args_string

            parsed = simple_parse_args_string(model_args)
            pretrained = parsed.get("pretrained", "OpenGVLab/InternVL3-9B")
            modality = parsed.get("modality", "image")
            device = parsed.get("device", "cuda:0")
            device_map = parsed.get("device_map", "cuda:0")
            batch_size = parsed.get("batch_size", "1")
            num_frame = int(parsed.get("num_frame", 32))
            num_layers = parsed.get("num_layers", None)
            if num_layers is not None:
                num_layers = int(num_layers)
            attn_implementation = parsed.get("attn_implementation", None)
            if attn_implementation in ("none", "None", ""):
                attn_implementation = None
            load_in_8bit = bool(parsed.get("load_in_8bit", False))
            image_max_num = int(parsed.get("image_max_num", 6))
            image_input_size = int(parsed.get("image_input_size", 448))
            clear_cuda_cache_each_step = bool(parsed.get("clear_cuda_cache_each_step", False))
        else:
            pretrained = "OpenGVLab/InternVL3-9B"
            modality = "image"
            device = "cuda:0"
            device_map = "cuda:0"
            batch_size = "1"
            num_frame = 32
            num_layers = None
            attn_implementation = None
            load_in_8bit = False
            image_max_num = 6
            image_input_size = 448
            clear_cuda_cache_each_step = False

        return cls(
            pretrained=pretrained,
            modality=modality,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            num_frame=num_frame,
            num_layers=num_layers,
            attn_implementation=attn_implementation,
            load_in_8bit=load_in_8bit,
            image_max_num=image_max_num,
            image_input_size=image_input_size,
            clear_cuda_cache_each_step=clear_cuda_cache_each_step,
            cfg=cfg,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

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

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def flatten(self, items):
        flattened = []
        for i in items:
            for j in i:
                flattened.append(j)
        return flattened

    def _safe_chat(self, pixel_values, question, gen_kwargs, num_patches_list=None):
        try:
            return self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                gen_kwargs,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
                gate_truthful_head=self.gate_truthful_head,
                truthful_head=self.truthful_head,
                hyperparams=self.hyperparams,
            )
        except TypeError:
            return self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                gen_kwargs,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v
            # Keep only supported generation args for model.chat
            for key in list(gen_kwargs.keys()):
                if key not in DEFAULT_GEN_KWARGS and key not in {"temperature", "top_p", "top_k", "min_p"}:
                    gen_kwargs.pop(key)

            context = contexts[0] if isinstance(contexts, list) else contexts
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if self.modality == "image":
                if visuals:
                    visuals = [
                        load_image(v, input_size=self.image_input_size, max_num=self.image_max_num).to(torch.bfloat16).cuda()
                        for v in visuals
                    ]
                    pixel_values = torch.cat(visuals, dim=0)
                    num_patches_list = [v.size(0) for v in visuals]
                    image_tokens = " ".join(["<image>"] * len(visuals))
                    question = image_tokens + "\n" + context
                else:
                    pixel_values = None
                    num_patches_list = None
                    question = context
                response, _history = self._safe_chat(pixel_values, question, gen_kwargs, num_patches_list=num_patches_list)
            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                pixel_values, num_patches_list = load_video(
                    visuals[0],
                    input_size=self.image_input_size,
                    max_num=self.image_max_num,
                    num_segments=self.num_frame,
                )
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                question = video_prefix + context
                response, _history = self._safe_chat(pixel_values, question, gen_kwargs, num_patches_list=num_patches_list)
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

            res.append(response)
            if self.clear_cuda_cache_each_step and torch.cuda.is_available():
                del pixel_values
                if self.modality == "image" and visuals:
                    del visuals
                gc.collect()
                torch.cuda.empty_cache()
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for InternVL3_9B")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("TODO: Implement multi-round generation for InternVL3_9B")
