import os
import glob
from absl import app, logging, flags

flags.DEFINE_string("movielens_dir", "./data/movielens", "path to the movielens folder")
FLAGS = flags.FLAGS

def main(_):
    pass

if __name__ == "__main__":
    app.run(main)