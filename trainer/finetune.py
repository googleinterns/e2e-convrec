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
"""Script for Running, Training, and Evaluating E2E Convrec Experiments."""

# SETUP
import functools
import os
import time
import warnings
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import t5
from contextlib import contextmanager
import logging as py_logging
import gzip
import json
import os
import tensorboard as tb
import random
import nltk
import sacrebleu
from absl import app
from absl import flags
from trainer import preprocessing, constants, metrics

FLAGS = flags.FLAGS
flags.DEFINE_integer('steps', 6000, "Finetuning training steps.")
flags.DEFINE_enum("size", "base", ["small", "base", "large", "3B", "11B"], "model size")
flags.DEFINE_string("name", "default", "name/description of model  version")
flags.DEFINE_enum("mode", "all", ["train", "evaluate", "all"], "run mode: train, evaluate, or all")
flags.DEFINE_integer("beam_size", 1, "beam size for saved model")
flags.DEFINE_float("temperature", 1.0, "temperature for saved model")
flags.DEFINE_float("learning_rate", .003, "learning rate for finetuning")
flags.DEFINE_string("tpu_topology", "2x2", "topology of tpy used for training")
def main(argv):
  """Main method for fintuning: builds, trains, and evaluates t5."""
  tf.disable_v2_behavior()
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  
  pretrained_dir = os.path.join(constants.BASE_PRETRAINED_DIR, FLAGS.size)
  model_dir = os.path.join(constants.MODELS_DIR, FLAGS.size, FLAGS.name)

  print("--------DETECTING TPUs----")
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    TPU_ADDRESS = tpu.get_master()
    print('Running on TPU:', TPU_ADDRESS)
  except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime')

  # load or build data
  if tf.io.gfile.exists(constants.RD_TSV_PATH["train"]) \
    and tf.io.gfile.exists(constants.RD_TSV_PATH["validation"]):
    print("TSV's Found")
    # Used cached data and counts.
    tf.logging.info("Loading Redial from cache.")
    num_rd_examples = json.load(tf.io.gfile.GFile(constants.RD_COUNTS_PATH))
  else:
    print("TSV's Not Found")
    # Create TSVs and get counts.
    tf.logging.info("Generating Redial TSVs.")
    num_rd_examples = {}
    for split, fname in constants.RD_SPLIT_FNAMES.items():
      print(os.path.join(constants.RD_JSONL_DIR, fname))
      num_rd_examples[split] = preprocessing.rd_jsonl_to_tsv(
          os.path.join(constants.RD_JSONL_DIR, fname), constants.RD_TSV_PATH[split])
    json.dump(num_rd_examples, tf.io.gfile.GFile(constants.RD_COUNTS_PATH, "w"))

  t5.data.TaskRegistry.add(
      "rd_recommendations",
      # Supply a function which returns a tf.data.Dataset.
      dataset_fn=preprocessing.rd_dataset_fn,
      splits=["train", "validation"],
      # Supply a function which preprocesses text from the tf.data.Dataset.
      text_preprocessor=[preprocessing.conversation_preprocessor],
      # Use the same vocabulary that we used for pre-training.
      sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
      # Lowercase targets before computing metrics.
      postprocess_fn=t5.data.postprocessors.lower_text, 
      # We'll use accuracy as our evaluation metric.
      metric_fns=[t5.evaluation.metrics.accuracy, t5.evaluation.metrics.bleu],
      # Not required, but helps for mixing and auto-caching.
      num_input_examples=num_rd_examples)

  # Set parallelism and batch size to fit on v2-8 TPU (if possible).
  # Limit number of checkpoints to fit within 5GB (if possible).
  model_parallelism, train_batch_size, keep_checkpoint_max = {
      "small": (1, 256, 16),
      "base": (2, 128, 8),
      "large": (8, 64, 4),
      "3B": (8, 16, 1),
      "11B": (8, 16, 1)}[FLAGS.size]

  tf.io.gfile.makedirs(model_dir)
  # The models from the t5 paper are based on the Mesh Tensorflow Transformer.
  model = t5.models.MtfModel(
      model_dir=model_dir,
      tpu=TPU_ADDRESS,
      tpu_topology=FLAGS.tpu_topology,
      model_parallelism=model_parallelism,
      batch_size=train_batch_size,
      sequence_length={"inputs": constants.INPUT_LENGTH, "targets": constants.TARGET_LENGTH},
      learning_rate_schedule=FLAGS.learning_rate,
      save_checkpoints_steps=2000,
      keep_checkpoint_max=keep_checkpoint_max,
  )

  if FLAGS.mode == "all" or FLAGS.mode == "train":
    model.finetune(
        mixture_or_task_name="rd_recommendations",
        pretrained_model_dir=pretrained_dir,
        finetune_steps=FLAGS.steps
    )

  # Evaluate and save predictions
  if FLAGS.mode == "all" or FLAGS.mode == "evaluate":
    model.batch_size = train_batch_size * 4 # a larger batch size requires less memory.
    model.eval(
        mixture_or_task_name="rd_recommendations",
        checkpoint_steps="all"
    )
    metrics.save_metrics("rd_recommendations", model_dir)


  # Export the SavedModel
  export_dir = os.path.join(model_dir, "export")

  model.batch_size = 1 # make one prediction per call
  saved_model_path = model.export(
      export_dir,
      checkpoint_step=-1,  # use most recent
      beam_size=FLAGS.beam_size,  # no beam search
      temperature=FLAGS.temperature,  # sample according to predicted distribution
  )
  print("Model saved to:", saved_model_path)

if __name__ == "__main__":
  app.run(main)