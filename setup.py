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
"""Sets up E2E Convrec module."""
from setuptools import find_packages
from setuptools import setup

setup(
    name='trainer',
    version='1.0',
    description='End to End Conversational Recommendations Experiments',
    
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        't5==0.8.0',
        'tensorflow-text==2.2',
        'nltk',
        'sacrebleu',
        'tensor2tensor'
    ],
    extras_require={
        'tensorflow': ['tensorflow==2.5.0'],
        'tensorflow_gpu': ['tensorflow-gpu==2.2.2'],
        'tensorflow-hub': ['tensorflow-hub>=0.6.0'],
    },
    python_requires='>=3.6',
)
