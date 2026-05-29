"""
Microsoft LLaVA-Med v1.5 (Mistral-7B): thin wrapper around :class:`~lmms_eval.models.llava.Llava`.

Hub: https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b

``generate_until``, ``loglikelihood``, etc. are not duplicated here: ``LlavaMed`` subclasses
``Llava`` and inherits all evaluation behavior; this module only overrides defaults
(``pretrained``, ``conv_template``, ``from_config``).

Loads via LLaVA-NeXT ``load_pretrained_model`` as ``LlavaMistralForCausalLM`` when the model name
contains ``mistral``. For local checkpoints, keep ``mistral`` in the directory name or pass
``model_name`` so the builder selects the Mistral branch.
"""

from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.llava import Llava, best_fit_attn_implementation

DEFAULT_PRETRAINED = "microsoft/llava-med-v1.5-mistral-7b"


@register_model("llava_med")
class LlavaMed(Llava):
    """
    Same behavior as ``Llava``; defaults match LLaVA-Med v1.5 (``mistral_instruct`` conv template).
    """

    def __init__(
        self,
        pretrained: str = DEFAULT_PRETRAINED,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="mistral_instruct",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,
        customized_config=None,
        cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            truncation=truncation,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            attn_implementation=attn_implementation,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            tie_weights=tie_weights,
            truncate_context=truncate_context,
            customized_config=customized_config,
            cfg=cfg,
            **kwargs,
        )

    @classmethod
    def from_config(cls, cfg, model_args=None):
        if model_args:
            from lmms_eval.utils import simple_parse_args_string

            parsed_model_args = simple_parse_args_string(model_args)
            pretrained = parsed_model_args.get("pretrained", DEFAULT_PRETRAINED)
            device = parsed_model_args.get("device", "cuda:0")
            batch_size = parsed_model_args.get("batch_size", 1)
            attn_implementation = parsed_model_args.get("attn_implementation", "eager")
            conv_template = parsed_model_args.get("conv_template", "mistral_instruct")
            use_cache = parsed_model_args.get("use_cache", True)
            truncate_context = parsed_model_args.get("truncate_context", False)
            model_name = parsed_model_args.get("model_name", None)
        else:
            pretrained = DEFAULT_PRETRAINED
            device = "cuda:0"
            batch_size = 1
            attn_implementation = "eager"
            conv_template = "mistral_instruct"
            use_cache = True
            truncate_context = False
            model_name = None

        return cls(
            pretrained=pretrained,
            device=device,
            batch_size=batch_size,
            attn_implementation=attn_implementation,
            conv_template=conv_template,
            use_cache=use_cache,
            truncate_context=truncate_context,
            model_name=model_name,
            cfg=cfg,
        )
