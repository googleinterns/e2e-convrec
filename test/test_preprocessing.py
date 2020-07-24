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
"""Unit Tests for the trainer.preprocessing scripts"""
import unittest
from trainer import preprocessing

def decode_example(ex):
  for key, value in ex.items():
    ex[key] = value.decode()

class TestPreprocessing(unittest.TestCase):
  """Class for testing the preprocessing functions"""

  def test_rd_preprocessing(self):
    """tests that the redial dataset is loaded properly"""
    # load the redial dataset
    ds = preprocessing.dataset_fn_wrapper("rd_recommendations")("validation")
    example = list(ds.take(5).as_numpy_iterator())[0]
    decode_example(example)

    # check format
    self.assertTrue("inputs" in example and "targets" in example,
                    ("the example should be loaded into "
                     "{'inputs': [], 'targets': []} format"))

    # load the preprocessed data
    ds_preprocessed = \
      preprocessing.preprocessor_wrapper("rd_recommendations")(ds)
    example = list(ds_preprocessed.take(5).as_numpy_iterator())[0]
    decode_example(example)

    # check that the data has been labeled witht the correct tag
    self.assertTrue("redial conversation:" in example["inputs"],
                    "inputs should be labeled with 'movielens sequence:'")

  def test_ml_seq_preprocessing(self):
    """tests that the ml sequences dataset is loaded properly"""
    # Load the movielens sequences dataset
    ds = preprocessing.dataset_fn_wrapper("ml_sequences")("validation")
    example = list(ds.take(5).as_numpy_iterator())[0]
    decode_example(example)

    # check format
    self.assertTrue("inputs" in example and "targets" in example,
                    ("the example should be loaded into "
                     "{'inputs': [], 'targets': []} format"))

    # load the preprocessed data
    ds_preprocessed = preprocessing.preprocessor_wrapper("ml_sequences")(ds)
    example = list(ds_preprocessed.take(5).as_numpy_iterator())[0]
    decode_example(example)

    # check that the data has been labeled witht the correct tag
    self.assertTrue("movielens sequence:" in example["inputs"],
                    "inputs should be labeled with 'movielens sequence:'")

  def test_ml_tags_preprocessing(self):
    """tests that the ml tags dataset is loaded properly"""
    # Load the movielens tags dataset
    ds = preprocessing.dataset_fn_wrapper("ml_tags_normal")("validation")
    example = list(ds.take(5).as_numpy_iterator())[0]
    decode_example(example)

    # check format
    self.assertTrue("inputs" in example and "targets" in example,
                    ("the example should be loaded into "
                     "{'inputs': [], 'targets': []} format"))

    # load the preprocessed data
    ds_preprocessed = preprocessing.preprocessor_wrapper("ml_tags")(ds)
    example = list(ds_preprocessed.take(5).as_numpy_iterator())[0]
    decode_example(example)

    # check that the data has been labeled witht the correct tag
    self.assertTrue("movielens tags:" in example["inputs"],
                    "inputs should be labeled with 'movielens tags:'")

  def test_ml_tags_format(self):
    """tests that the ml tags dataset can be loaded in different formats"""
    # load normal
    ds = preprocessing.dataset_fn_wrapper("ml_tags_normal")("validation")
    normal = list(ds.take(5).as_numpy_iterator())[0]
    decode_example(normal)

    # load reversed
    ds = preprocessing.dataset_fn_wrapper("ml_tags_reversed")("validation")
    reverse = list(ds.take(5).as_numpy_iterator())[0]
    decode_example(reverse)

    # load masked
    ds = preprocessing.dataset_fn_wrapper("ml_tags_masked")("validation")
    masked = list(ds.take(5).as_numpy_iterator())[0]
    decode_example(masked)

    # test reversing
    self.assertEqual(normal["inputs"], reverse["targets"],
                     "testing switched inputs and targets")
    self.assertEqual(normal["targets"], reverse["inputs"],
                     "testing switched inputs and targets")

    # test masking
    self.assertTrue("<extra_id" in masked["inputs"],
                    "masked should contain mask tokens")

if __name__ == '__main__':
  unittest.main()
