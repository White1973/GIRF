import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIPE_DATA_ROOT = Path(
    os.environ.get("DIPE_DATA_ROOT", PROJECT_ROOT.parent / "DIPE" / "data")
).resolve()
CONTROL_DATA_ROOT = Path(
    os.environ.get(
        "MULTI_IMAGE_CONTROL_ROOT", PROJECT_ROOT / "data" / "multi_image_control_v1"
    )
).resolve()

LLAVA_PRETRAIN = {
    "annotation_path": str(
        DIPE_DATA_ROOT / "llava_pretrain_data" / "blip_laion_cc_sbu_558k.json"
    ),
    "data_path": str(DIPE_DATA_ROOT / "llava_pretrain_data"),
}

LLAVA_NEXT = {
    "annotation_path": str(
        DIPE_DATA_ROOT
        / "llava_next_data"
        / "llava_next_raw_format"
        / "llava_next_raw_format_processed.json"
    ),
    "data_path": str(DIPE_DATA_ROOT / "llava_next_data" / "llava_next_raw_format"),
}

MULTI_IMAGE_CONTROL = {
    "annotation_path": str(CONTROL_DATA_ROOT / "train.json"),
    # The generated manifest prefixes each path with llava_next_data/ or
    # llava_pretrain_data/, so both source datasets share this media root.
    "data_path": str(DIPE_DATA_ROOT),
}

MULTI_IMAGE_CONTROL_SMOKE = {
    "annotation_path": str(CONTROL_DATA_ROOT / "smoke.json"),
    "data_path": str(DIPE_DATA_ROOT),
}

MULTI_IMAGE_CONTROL_VALIDATION = {
    "annotation_path": str(CONTROL_DATA_ROOT / "validation.json"),
    "data_path": str(DIPE_DATA_ROOT),
}

data_dict = {
    "llava_pretrain": LLAVA_PRETRAIN,
    "llava_next_sft": LLAVA_NEXT,
    "multi_image_control_v1": MULTI_IMAGE_CONTROL,
    "multi_image_control_v1_smoke": MULTI_IMAGE_CONTROL_SMOKE,
    "multi_image_control_v1_validation": MULTI_IMAGE_CONTROL_VALIDATION,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["multi_image_control_v1"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
