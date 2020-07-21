# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared Constants."""
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
RD_COUNTS_PATH = os.path.join(DATA_DIR, "rd-counts.json")
RD_TSV_PATH = {
    "train": os.path.join(DATA_DIR, "rd-train.tsv"),
    "validation": os.path.join(DATA_DIR, "rd-validation.tsv")
}
ML_SEQ_TSV_PATH = {
    "train": os.path.join(DATA_DIR, "ml-train.tsv"),
    "validation": os.path.join(DATA_DIR, "ml-validation.tsv")
}
ML_TAGS_TSV_PATH = {
    "train": os.path.join(DATA_DIR, "ml-tags-train.tsv"),
    "validation": os.path.join(DATA_DIR, "ml-tags-test.tsv")
}