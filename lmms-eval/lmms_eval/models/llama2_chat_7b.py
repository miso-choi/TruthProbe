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
        # self._model = AutoModelForCausalLM.from_pretrained(pretrained,device_map=self._device, attn_implementation=attn_implementation)

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

        #self._image_processor = InstructBlipProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, attn_implementation=attn_implementation)
        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
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

        for contexts, doc_to_target,doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
        # 정답 continuation 생성
            if isinstance(doc_to_target, str):
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            try:
                continuation = int(continuation)
            except (ValueError, TypeError):
                pass
            # 프롬프트 만들기

            prompts_text = contexts[0] if isinstance(contexts, list) else contexts

            # 전체 입력 = prompt + 정답 continuation
            full_input = prompts_text + str(continuation)

            # tokenizing
            input_ids = self.tokenizer(full_input, return_tensors="pt").input_ids.to(self.device)

            # context 길이 측정
            context_len = len(self.tokenizer(prompts_text)["input_ids"])

            labels = input_ids.clone()
            labels[:, :context_len] = -100  # context에 해당하는 부분은 마스킹

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, use_cache=self.use_cache)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]

            pred_tokens = logits.argmax(dim=-1)
            target_tokens = input_ids[:, context_len:]
            pred_cont = pred_tokens[:, context_len:]
            # continuation이 모델이 정답으로 평가하는 부분
            #predicted_text = str(continuation).strip()

            # 해당 doc_id에 맞는 Instance 객체 찾아서 output 설정
            #for instance in requests:
            #    if instance.args[3] == doc_id:  # doc_id 매칭
            #        instance.output = predicted_text  # ✅ 로그 저장용
            #        break


            is_greedy = (pred_cont == target_tokens).all()
            res.append((loss.item(), bool(is_greedy)))
            #res.append(continuation)
            #print(continuation)
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
            #visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            #visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            #context = contexts[0]

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
            
            #이미지 부분 삭제
            #inputs = self._tokenizer(context, return_tensors="pt", truncation=True).to(self.device)
            # 기존 코드: context = contexts[0]
            
            #texts = [self._tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            # Vicuna-style 프롬프트 생성
            #chat_prompt = (
            #    f"### Human: {context}\n"
            #    "### Assistant: "
            #)


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
                    # temperature=gen_kwargs["temperature"],
                    # top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=max_new_tokens, # gen_kwargs["max_new_tokens"]
                    use_cache=self.use_cache,
                    # Custom parameters for faithful evaluation
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


    def generate_until_with_head_masking(self, requests: List[Instance]) -> List[str]:
        """
        Process samples individually to avoid OOM issues.
        Writes results incrementally to JSONL file and returns minimal response for framework compatibility.
        """
        import json
        import os
        from datetime import datetime
        
        # Create output directory and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get output path from configuration or use default
        if self.output_dir:
            base_output_dir = self.output_dir
        else:
            # Fallback to current working directory
            base_output_dir = os.getcwd()
        
        # Create subdirectory for attention difference results
        output_dir = os.path.join(base_output_dir, "attn_diff_results/vicuna-7b-v1.5")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"attn_diff_results_{timestamp}.jsonl")
        
        print(f"Starting attention difference analysis for {len(requests)} samples...")
        print(f"Results will be saved to: {output_file}")
        
        # Process each sample individually to avoid memory accumulation
        with torch.no_grad():
            for request in tqdm(requests, total=len(requests), desc="Processing samples"):
                # Extract sample information
                contexts, doc_to_target, doc_to_visual, doc_id, task, split = request.args
                sample_data = self.task_dict[task][split][doc_id]
                
                # Get image path and text input
                image_path = sample_data.get('image_path', '')
                text_input = contexts[0] if isinstance(contexts, list) else contexts
                pos_target_word = sample_data.get('positive_target_word', '')
                neg_target_word = sample_data.get('negative_target_word', '')
                    
                try:
                    # Create single-sample requests
                    single_pos_requests = self.update_samples_with_target_word([request], 'positive')
                    single_neg_requests = self.update_samples_with_target_word([request], 'negative')
                    
                    # Get attention from original model for this sample
                    mask_info = None
                    _, pos_attn = self.generate_until(single_pos_requests, mask_info)
                    _, neg_attn = self.generate_until(single_neg_requests, mask_info)

                    
                    # Get attention difference from original model
                    original_attn_diff, original_pos_attn, original_neg_attn = self.get_attn_diff(pos_attn, neg_attn)
                    
                    # Store attention differences for this sample
                    attn_diff_results = {}
                    attn_diff_results['original'] = {
                        'attn_diff': original_attn_diff[0].item(),
                        'pos_attn': original_pos_attn[0].item(),
                        'neg_attn': original_neg_attn[0].item(),
                    }
                    
                    
                    ablation_count = 0
                    for layer in tqdm(range(self.head_masking_start_layer, self.head_masking_end_layer + 1), desc=f"Sample {doc_id} - Layers"):
                        for head in tqdm(range(self.head_masking_start_head, self.head_masking_end_head + 1), desc=f"Sample {doc_id} - Heads"):
                            ablation_count += 1
                            
                            mask_info = self.update_mask_info(layer, head)
                            
                            # Get attention from ablated model for this sample
                            _, pos_attn = self.generate_until(single_pos_requests, mask_info)
                            _, neg_attn = self.generate_until(single_neg_requests, mask_info)
                            
                            # Get attention difference
                            ablated_attn_diff, ablated_pos_attn, ablated_neg_attn = self.get_attn_diff(pos_attn, neg_attn)
                            
                            attn_diff_results[(layer, head)] = {
                                'attn_diff': ablated_attn_diff[0].item(),
                                'pos_attn': ablated_pos_attn[0].item(),
                                'neg_attn': ablated_neg_attn[0].item()
                            }
                            

                    # Sort attention differences by magnitude (most important first) - ONLY ablated results
                    # Filter out 'original' from sorting
                    ablated_items = [(k, v) for k, v in attn_diff_results.items() if k != 'original']
                    sorted_ablated_attn_diff = sorted(
                        ablated_items, 
                        key=lambda x: x[1]['attn_diff'], 
                        reverse=True
                    )
                    
                    # Take only top-10
                    # top_10_sorted_attn_diff = sorted_ablated_attn_diff[:10]
                    # Take all ablated results
                    sorted_attn_diff = sorted_ablated_attn_diff
                    del sorted_ablated_attn_diff, ablated_items
                    
                    # Create the final result for this sample
                    sample_result = {
                        "image_path": image_path,
                        "text_input": text_input,
                        "pos_target_word": pos_target_word,
                        "neg_target_word": neg_target_word,
                        # "sorted_attn_diff": top_10_sorted_attn_diff,  # Only top-10 ablated results
                        "sorted_attn_diff": sorted_attn_diff, # All ablated results
                        "original_attn_diff": attn_diff_results['original'],  # Keep original separate
                        "doc_id": doc_id,
                        "timestamp": timestamp,
                        "output_file": output_file
                    }
                    
                    # Write result immediately to JSONL file
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')
                    
                    # Clear GPU memory after processing each sample
                    del sample_result
                    print(f"Sample {doc_id}: Completed attention difference analysis - saved to {output_file}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"❌ OOM ERROR on Sample {doc_id}!")
                        print(f"❌ Image path: {image_path}")
                        print(f"❌ Text length: {len(text_input)} chars")
                        if torch.cuda.is_available():
                            memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                            print(f"❌ GPU memory at OOM: {memory_after:.2f} GB")
                        # Continue with next sample instead of crashing
                        continue
                    else:
                        raise e
        
        print(f"Completed attention difference analysis for all {len(requests)} samples!")
        print(f"All results saved to: {output_file}")
        
        # Return minimal response for framework compatibility
        # Just return a simple acknowledgment since results are already saved to file
        return [json.dumps({"status": "completed", "output_file": output_file, "num_samples": len(requests)})]

    def get_attn_diff(self, pos_attn, neg_attn):
        """
        Calculate attention difference for batch processing.
        Returns attention differences for all samples in the batch.
        """
        
        attn_diff = neg_attn - pos_attn  # Shape: (bs,)
        
        return attn_diff, pos_attn, neg_attn
    
    def update_mask_info(self, layer, head):

        if self.head_masking:
            mask_qkv = self.mask_qkv
            mask_scale_factor = self.mask_scale_factor
            
            # Convert string to float if needed
            if isinstance(mask_scale_factor, str):
                mask_scale_factor = float(mask_scale_factor)

            mask_info = {
                'layer': layer, 
                'head': head,
                'mask_qkv': mask_qkv,
                'mask_scale_factor': mask_scale_factor,
                }

        return mask_info


    def update_samples_with_target_word(self, requests: List[Instance], target_type: str) -> List[Instance]:
        """
        Create new requests with target word replaced.
        target_type: 'positive' or 'negative'
        """
        updated_requests = []
        
        for request in requests:
            # Extract the original arguments
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = request.args
            
            # Get the original sample data
            sample_data = self.task_dict[task][split][doc_id]
            
            # Get the target words
            if target_type == 'positive':
                target_word = sample_data.get('positive_target_word', '')
            else:  # negative
                target_word = sample_data.get('negative_target_word', '')
            
            # Replace {target word} in the context
            if isinstance(contexts, list):
                updated_contexts = [ctx.replace('{target word}', target_word) for ctx in contexts]
            else:
                updated_contexts = contexts.replace('{target word}', target_word)
            
            # Create new Instance with updated arguments
            updated_args = (updated_contexts, doc_to_target, doc_to_visual, doc_id, task, split)
            updated_request = Instance(
                request_type=request.request_type,
                arguments=updated_args,
                idx=request.idx,
                metadata={"task": task, "doc_id": doc_id, "repeats": request.repeats}
            )
            updated_requests.append(updated_request)
        
        return updated_requests

    def generate_until_with_save_attention(self, requests: List[Instance]) -> List[str]:
        """
        Process samples individually to avoid OOM issues.
        Writes results incrementally to JSONL file and returns minimal response for framework compatibility.
        """
        import json
        import os
        from datetime import datetime
        
        # Create output directory and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get output path from configuration or use default
        if self.cfg.get('metadata').get("output_dir", None):
            base_output_dir = self.cfg.get('metadata').get("output_dir")
        else:
            # Fallback to current working directory
            base_output_dir = os.getcwd()
        
        # Create subdirectory for attention difference results
        output_dir = os.path.join(base_output_dir, "save_attention_results/vicuna-7b-v1.5")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"save_attention_results_{timestamp}.jsonl")
        
        print(f"Starting save attention results for {len(requests)} samples...")
        print(f"Results will be saved to: {output_file}")
        
        # Process each sample individually to avoid memory accumulation
        with torch.no_grad():
            for request in tqdm(requests, total=len(requests), desc="Processing samples"):
                # Extract sample information
                contexts, doc_to_target, doc_to_visual, doc_id, task, split = request.args
                sample_data = self.task_dict[task][split][doc_id]
                
                # Get image path and text input
                image_path = sample_data.get('image_path', '')
                text_input = contexts[0] if isinstance(contexts, list) else contexts
                pos_target_word = sample_data.get('positive_target_word', '')
                neg_target_word = sample_data.get('negative_target_word', '')

                print(f"Processing sample {doc_id} of {len(requests)}")
                    
                try:
                    # Create single-sample requests
                    single_pos_requests = self.update_samples_with_target_word([request], 'positive')
                    single_neg_requests = self.update_samples_with_target_word([request], 'negative')
                    
                    # Get attention from original model for this sample
                    mask_info = None
                    _, pos_attn = self.generate_until(single_pos_requests, mask_info)

                    # Save tensors using torch.save()
                    pt_output_file = os.path.join(output_dir, f"samples/pos_attn/doc_id_{doc_id}.pt")
                    if not os.path.exists(pt_output_file):
                        os.makedirs(os.path.dirname(pt_output_file), exist_ok=True)
                    torch.save(pos_attn, pt_output_file)
                    del pos_attn

                    _, neg_attn = self.generate_until(single_neg_requests, mask_info)
                    pt_output_file = os.path.join(output_dir, f"samples/neg_attn/doc_id_{doc_id}.pt")
                    if not os.path.exists(pt_output_file):
                        os.makedirs(os.path.dirname(pt_output_file), exist_ok=True)
                    torch.save(neg_attn, pt_output_file)
                    del neg_attn
                    
                    # Create the final result for this sample
                    meta_data = {
                        "doc_id": doc_id,
                        "image_path": image_path,
                        "text_input": text_input,
                        "pos_target_word": pos_target_word,
                        "neg_target_word": neg_target_word,
                        # "pos_attn": pos_attn_np,
                        # "neg_attn": neg_attn_np,
                        "timestamp": timestamp,
                    }
                    
                    # Write result immediately to JSONL file
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(meta_data, ensure_ascii=False) + '\n')
                    
                    # Clear GPU memory after processing each sample
                    del meta_data
                    print(f"Sample {doc_id}: Completed save attention results - saved to {output_file}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"❌ OOM ERROR on Sample {doc_id}!")
                        print(f"❌ Image path: {image_path}")
                        print(f"❌ Text length: {len(text_input)} chars")
                        if torch.cuda.is_available():
                            memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                            print(f"❌ GPU memory at OOM: {memory_after:.2f} GB")
                        # Continue with next sample instead of crashing
                        continue
                    else:
                        raise e
        
        print(f"Completed attention difference analysis for all {len(requests)} samples!")
        print(f"All results saved to: {output_file}")
        
        # Return minimal response for framework compatibility
        # Just return a simple acknowledgment since results are already saved to file
        return [json.dumps({"status": "completed", "output_file": output_file, "num_samples": len(requests)})]


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
        # Extract model-specific config from the task config
        # You can access task-specific parameters here
        generation_kwargs = cfg.get("generation_kwargs", {})
        
        # Parse model arguments if provided
        if model_args:
            from lmms_eval.utils import simple_parse_args_string
            parsed_model_args = simple_parse_args_string(model_args)
            pretrained = parsed_model_args.get("pretrained", "lmsys/vicuna-7b-v1.5")
            device = parsed_model_args.get("device", "cuda:0")
            batch_size = parsed_model_args.get("batch_size", 1)
            # attn_implementation = parsed_model_args.get("attn_implementation", "eager")
            # conv_template = parsed_model_args.get("conv_template", "vicuna_v1")
            use_cache = parsed_model_args.get("use_cache", True)
            # truncate_context = parsed_model_args.get("truncate_context", False)
        else:
            pretrained = "lmsys/vicuna-7b-v1.5"
            device = "cuda:0"
            batch_size = 1
            # attn_implementation = "eager"
            # conv_template = "vicuna_v1"
            use_cache = True
            # truncate_context = False
        
        return cls(
            pretrained=pretrained,  # Extract from model_args instead of hardcoding
            device=device,
            batch_size=batch_size,
            # attn_implementation=attn_implementation,
            # conv_template=conv_template,
            use_cache=use_cache,
            # truncate_context=truncate_context,
            cfg=cfg,  # Pass the full task config
        )
