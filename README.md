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

## Download the Grouplens MovieLens dataset

The dataset can be found at https://grouplens.org/datasets/movielens/25m/

Download it, extract the ml-25m folder and move it into data/movielens

## Prepare MovieLens users' watch sequences

The script `generate_movielens_user_dialogs.py` can be used to generate sequences of liked movies for each user. To run this script, first you need to download and install the [Protocol buffer compiler](https://developers.google.com/protocol-buffers/docs/downloads). 

Then, compile the proto messages using
```
protoc -I=. --python_out=.  movie_lens_rating.proto 
protoc -I=. --python_out=.  dialog.proto 
```

After that run the following command.
```
python3 generate_movielens_user_dialogs.py \
--ratings_file_path=data/movielens/ml-25m/ratings.csv \
--movies_dict_path=data/movielens/ml-25m/movies.csv \
--num_ratings_per_user=10 \
--liked_threshold=4.0 \
--output_seq_path=data/movielens/sequences/user_watch_seq.csv \
--output_dialog_path=data/movielens/sequences/user_dialogs.tfrecord \
--num_shards=5
```


## Running the project:

To post a job, you need to use the gcloud jobs submit training command to post a job using the
module located in trainer.fintune:

    PROJECT_NAME=$USER_test_job && \
    gcloud ai-platform jobs submit training $PROJECT_NAME \
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
    --name=quickstart \
    --mode=all \
    --task=combined

### Important flags:

`--steps`: controls the number of finetuning steps

`--size`: controls the model size (`small`, `base`, `large`, `3B`, or `11B`)

`PROJECT_NAME` is a unique identifier to the job instance. You can find the existing/used names in:
 `gcloud ai-platform jobs list`
 `--mode`: determines the run mode. There are eight options:
 + `rd_recommendations`: dialogue responses from the redial dataset

+ `train`: train a model

+ `evaluate`: evaluate model using standard metrics

+ `export`: export a checkpoint

+ `all`: run training, evaluation, and export the last model

+ `probe_1`: run probe 1

+ `probe_2`: run probe 2

+ `probe_3`: run probe 3

+ `probe_4`: run probe 4

`--task`: determines which training tasks are run. There are three tasks:

+ `rd_recommendations`: dialogue responses from the redial dataset

+ `ml_sequences`: sequences of movies from the movielens dataset

+ `ml_tags`: tags to movie mappings from the movielens dataset

+ `ml_reviews`: movie to review mappings from the movielens and IMDB datasets

+ Set the `--task` flag to equal any of these options:

  + `rd_recommendations`, `ml_tags`, `ml_sequences`, `ml_reviews`, `ml_all` (seqs + tags + reviews), `rd_tags` (redial + tags), `rd_sequences` (redial + seqs), `rd_reviews` (redial + reviews), `combined` (all four)
    gcloud ai-platform jobs list


## Running tensorboard:

If you have access to the gcloud bucket, you can start tensorboard by connecting to the bucket (you'll have to update the logdir path):

`tensorboard --logdir=gs://e2e_central/models/base --port=8080`

## Setting Up Dev Enviroment:

if you want to set up a dev enviroment with the right dependencies installed, you can create a virtual enviroment and install the requirements.txt. Any type of virtual enviroment should work.

Example:

`cd ~ && git clone git@github.com:googleinterns/e2e-convrec.git`

`python3 -m venv ~/e2e-convrec`

`cd e2e-convrec`

`source bin/activate`

`pip3 install -r requirements.txt`


## Rebuilding the training data:

if you wanted to reformat the training data you can use `python3 -m data.build_redial` to run the script to format the redial dataset. This shouldn't be necessary (the data is already formatted).

## To Run Tests

You can use the command `nosetests -a '!network'` to run all local tests or `nosetests to run all tests`, or `nosetests <specific_test.py>` to run a specific test

## Troubleshooting

### I've see an error: 'Anonymous caller does not have storage.objects.get access to the Google Cloud Storage object.'

After installing gcloud SDK, you may need to authenticate to generate credential files for command line access and SDK access:

    `gcloud auth application-default login`
    `gcloud auth login`

The latter, when executed in this order, generate credentials for SDK login.