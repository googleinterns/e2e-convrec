import os

INPUT_LENGTH = 512 #1033 - longest training input for redial
TARGET_LENGTH = 128 #159 - longest trainin target for redial
BASE_DIR = "gs://e2e_central"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
RD_JSONL_DIR = "gs://e2e_central/data/redial/"
RD_SPLIT_FNAMES = {
    "train": "rd-train-formatted.jsonl",
    "validation": "rd-test-formatted.jsonl"
}
rd_counts_path = os.path.join(DATA_DIR, "rd-counts.json")
rd_tsv_path = {
    "train": os.path.join(DATA_DIR, "rd-train.tsv"),
    "validation": os.path.join(DATA_DIR, "rd-validation.tsv")
}
