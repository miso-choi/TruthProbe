from lmms_eval.api.task import ConfigurableTask
from lmms_eval.tasks.rlhf_v_aug.utils import attn_diff_process_results


class AttnDiffTask(ConfigurableTask):
    """
    Custom task class for attention difference analysis.
    Overrides process_results to handle dictionary results from generate_until_with_head_masking.
    Also provides custom logging to exclude redundant data.
    """
    
    def process_results(self, doc, results):
        """
        Override the default process_results to handle attention difference results.
        The results from generate_until_with_head_masking are JSON strings.
        """
        # Call our custom processing function directly
        return attn_diff_process_results(doc, results)
    
    def get_clean_logging_data(self, doc_id, doc, target, requests, filter_key, metrics):
        """
        Override logging to exclude resps and filtered_resps for clean results.
        """
        import json
        from lmms_eval.utils import hash_string, handle_non_serializable
        
        # Create clean document (exclude image data)
        saved_doc = {}
        for key, value in doc.items():
            if "image" not in key:
                if isinstance(value, dict) and "array" in value:
                    continue
                else:
                    saved_doc[key] = value
        
        # Filter arguments
        filtered_arguments = []
        for req in requests:
            for value in req.args:
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    filtered_arguments.append(value)
        
        # Create clean example without resps and filtered_resps
        example = {
            "doc_id": doc_id,
            "doc": saved_doc,
            "target": target,
            "arguments": filtered_arguments,
            # Exclude resps and filtered_resps to avoid redundancy
            # "resps": [req.resps for req in requests],  # REMOVED
            # "filtered_resps": [req.filtered_resps[filter_key] for req in requests],  # REMOVED
            "doc_hash": hash_string(
                json.dumps(
                    requests[0].doc,
                    indent=2,
                    default=handle_non_serializable,
                    ensure_ascii=False,
                )
            ),
            "prompt_hash": hash_string(requests[0].arguments[0]),
            "target_hash": hash_string(str(target)),
        }
        
        # Add metrics (which include attn_diff_data)
        example.update(metrics)
        
        return example 