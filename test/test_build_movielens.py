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

"""Unit Tests for E2E Convrec modules"""
from data import build_movielens
import unittest

class TestBuildMovielens(unittest.TestCase):

    def test_flip_titles(self):
        test_inputs = [
            "Green Mile, The (1999) @ Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966) @ Devil's Advocate, The (1997) ",
            "King's Speech, The (2010) @ Social Network, The (2010) @ Catch Me If You Can (2002)",
            "Brady Bunch Movie, The (1995) @ Shining, The (1980) @ Cool Hand Luke (1967)",
            "House Bunny, The (2008)",
            "Ten Commandments, The (1956)",
            "Fake Movie, the (subtitle) weirdness, (0000)"
        ]

        test_outputs = [
            "The Green Mile (1999) @ The Good, the Bad and the Ugly (Buono, il brutto, il cattivo, Il) (1966) @ The Devil's Advocate (1997)",
            "The King's Speech (2010) @ The Social Network (2010) @ Catch Me If You Can (2002)",
            "The Brady Bunch Movie (1995) @ The Shining (1980) @ Cool Hand Luke (1967)",
            "The House Bunny (2008)",
            "The Ten Commandments (1956)",
            "the Fake Movie (subtitle) weirdness, (0000)"
        ]
        for test_input, test_output in zip(test_inputs, test_outputs):
            print(build_movielens.flip_titles(test_input), test_output)
            self.assertEqual(build_movielens.flip_titles(test_input), test_output, "should put title in order")

if __name__ == '__main__':
    unittest.main()