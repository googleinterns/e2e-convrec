from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import apache_beam as beam
import pandas as pd

import dialog_pb2
import movie_lens_rating_pb2 as rating_pb2


def encode_line_as_rating_proto(line):
  """ Parses a comma separated value of movielens rating as MovieLensRating proto message"""
  tokens = line.split(",")
  rating = rating_pb2.MovieLensRating(
      user_id=int(tokens[0]),
      movie_id=int(tokens[1]),
      rating=float(tokens[2]),
      timestamp=int(tokens[3]))
  return rating


def map_seq_to_dialog(user_id_and_watch_seq, movies_dict,
                      substitute_movie_id_with_title):
  """ Generates a Dialog proto message from the given sequence of liked movies. """
  user_id = user_id_and_watch_seq[0]
  watch_seq = user_id_and_watch_seq[1]
  if substitute_movie_id_with_title:
    input_movie_titles_seq = " @ ".join(
        movies_dict[int(id)] for id in watch_seq[:-1])
    target_movie_title = movies_dict[int(watch_seq[-1])]
  else:
    input_movie_titles_seq = " ".join("@" + str(id) for id in watch_seq[:-1])
    target_movie_title = "@" + str(watch_seq[-1])
  dialog_events = []
  dialog_events.append(
      dialog_pb2.DialogEvent(
          speaker="USER",
          utterance=input_movie_titles_seq.encode("utf-8"),
          time_ms=1))
  dialog_events.append(
      dialog_pb2.DialogEvent(
          speaker="ASSISTANT",
          utterance=target_movie_title.encode("utf-8"),
          time_ms=2))
  dialog = dialog_pb2.Dialog(
      source="conversation_{}".format(user_id), events=dialog_events)
  return dialog


def create_movies_dict(movies_path):
  """ returns a dictionary of movie Id to movie title."""
  with open(movies_path) as fh:
    movies_df = pd.read_csv(fh)
    movies_df = movies_df.set_index(["movieId"])
    return movies_df["title"].to_dict()


def create_pipeline(input_ratings_path, movies_dict_path, liked_threshold,
                    num_ratings_per_user, substitute_movie_id_with_title):
  movies_dict = create_movies_dict(movies_dict_path)

  def pipeline(root):
    user_watch_seq = (
        root
        | "CreateRatings" >> beam.io.ReadFromText(
            input_ratings_path, skip_header_lines=1)
        | "EncodeAsProto" >>
        beam.Map(lambda line: encode_line_as_rating_proto(line))
        | "FilterByRatings" >>
        beam.Filter(lambda rating: rating.rating >= liked_threshold)
        | "SetUserIdAsKey" >> beam.Map(lambda rating: (rating.user_id, rating))
        | "GroupByUser" >> beam.GroupByKey()
        | "ConvertValueToList" >> beam.Map(lambda kv: (kv[0], list(kv[1])))
        | "FilterByUserRatingsCount" >>
        beam.Filter(lambda kv: len(kv[1]) >= num_ratings_per_user)
        | "SortRatingsByTimestamp" >> beam.Map(lambda kv: (kv[
            0
        ], list(sorted(kv[1], key=lambda rating_proto: rating_proto.timestamp)))
                                              )
        | "KeepLatestRatings" >>
        beam.Map(lambda kv: (kv[0], kv[1][-num_ratings_per_user:]))
        | "KeepOnlyMovieIdList" >>
        beam.Map(lambda kv: (kv[0], [rating.movie_id for rating in kv[1]])))
    user_watch_dialog = (
        user_watch_seq
        | "WatchSeqToDialog" >> beam.Map(map_seq_to_dialog, movies_dict,
                                         substitute_movie_id_with_title))
    return user_watch_seq, user_watch_dialog
  return pipeline