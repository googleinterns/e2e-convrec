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
"""Unit Tests for build redial script"""
import unittest
from absl import logging
from data import build_redial

class TestPreprocessing(unittest.TestCase):
  def test_separate_responses(self):
    test_data_files = {
        "inputs": "./test/test-data/separate_responses_inputs.jsonl",
        "outputs": "./test/test-data/separate_responses_outputs.jsonl"}
    test_inputs = build_redial.read_jsonl(test_data_files["inputs"])
    test_outputs = build_redial.read_jsonl(test_data_files["outputs"])
    self.assertEqual(build_redial.separate_responses(test_inputs), test_outputs)
    logging.info(build_redial.separate_responses(test_inputs))

if __name__ == '__main__':
  unittest.main()
