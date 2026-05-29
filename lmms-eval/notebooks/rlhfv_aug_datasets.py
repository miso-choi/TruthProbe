import os
import json
import datasets

_CITATION = "RLHF‑v‑augmented dataset"
_DESCRIPTION = """
Dataset built from a single annotations.json reference, loading only images listed under `image_path`.
Fields: idx, text_input, negative_target_word, positive_target_word, image.
"""

class RLHFVAugDataset(datasets.GeneratorBasedBuilder):
    """Loads images and key‑feature annotations from annotations.json, no train/validation split."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "idx": datasets.Value("int32"),
                "text_input": datasets.Value("string"),
                "negative_target_word": datasets.Value("string"),
                "positive_target_word": datasets.Value("string"),
                "image": datasets.Image(),
            }),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager.download_and_extract(".") returns repository root in local env
        repo_path = dl_manager.download_and_extract('.') or "."
        ann_path = os.path.join(repo_path, "annotations.json")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"annotations.json not found at {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        # single "full" split covering all annotated items
        return [
            datasets.SplitGenerator(
                name="all",
                gen_kwargs={"annotations": data_list, "repo_path": repo_path}
            ),
        ]

    def _generate_examples(self, annotations, repo_path):
        images_root = os.path.join(repo_path, "images")
        seen = set()

        for idx, ann in enumerate(annotations):
            key = str(ann.get("idx", idx))
            if key in seen:
                continue
            seen.add(key)

            rel_path = ann.get("image_path", "")
            abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(images_root, rel_path)
            if not os.path.isfile(abs_path):
                # 파일이 없으면 사용자에게 한마디 로그 남기고 skip
                # print(f"Warning: {abs_path} does not exist, skipping {key}")
                continue

            yield key, {
                "idx": ann.get("idx", idx),
                "text_input": ann.get("text_input", ""),
                "negative_target_word": ann.get("negative_target_word", ""),
                "positive_target_word": ann.get("positive_target_word", ""),
                "image": abs_path,
            }
