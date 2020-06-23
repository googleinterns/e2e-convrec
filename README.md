**This is not an officially supported Google product.**

Repository for conversational recommender intern research project.

## Source Code Headers

Every file containing source code must include copyright and license
information. This includes any JS/CSS files that you might be serving out to
browsers. (This is to help well-intentioned people avoid accidental copying that
doesn't comply with the license.)

Apache header:

    Copyright 2020 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

## Running the project:

To post a job, you need to use the gcloud jobs submit training command to post a job using the
module located in trainer.fintune:

    gcloud ai-platform jobs submit training new_default \
    --staging-bucket gs://e2e_central \
    --package-path ./trainer \
    --module-name trainer.finetune \
    --region us-central1 \
    --runtime-version=2.1 \
    --python-version=3.7 \
    --scale-tier=BASIC_TPU \
    -- \
    --steps=6000 \
    --size=base \
    --name=quickstart
    --mode=all

steps, size, name, and mode are user flags which default to 6000, base, default, and all. Size has to be either small, base, large, 3B, or 11B. Mode can be set to train
(for finetuning), evaluation (for evaulating the most recnt checkpoint), or all (for both). if there is already a model at gs://e2e_central/models/$SIZE/$NAME, 
the program will continue training from that model's most recent checkpoint

## Running tensorboard:

If you have access to the gcloud bucket, you can start tensorboard by connecting to the bucket (you'll have to update the logdir path):

`tensorboard --logdir=gs://e2e_central/models/base --port=8080`

## Rebuilding the training data:

if you wanbted to reformat the training data you can use `python3 -m data.build_redial` to run the script to format the redial dataset. This shouldn't be necessary (the data is already formatted).
