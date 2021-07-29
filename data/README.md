# Data for the E2E Convrec Project

  

  

`data/redial`contains the redial dataset as unpacked from [the redial website]
(https://redialdata.github.io/website/) (`rd-test-data.jsonl`, `rd-train-data.jsonl`) as well as the 
version that has been reformatted for training (`rd-train-formatted.jsonl`, `rd-test-
formatted.jsonl`).

  

  

`build_redial.py` contains the script to generate the formatted data from the dataset. There should 
be no need to rerun the `build_redial.py` script unless it is necessary to rebuild the data with a 
different format.

  

  

You can run the `build_redial.py` script using `python3 -m data.build_redial` from the project 
root folder.

  

If you download the MovieLens 25m dataset from [the GroupLens website]
(https://grouplens.org/datasets/movielens/) into `data/movielens/ml-25m`, you can run the 
`generate_movielens_user_dialogs.py` (more instructions in the project README) to generate the 
MovieLens sequences in the `/data/movielens/sequences` folder. Then you can run `python3 -m 
data.build_movielens` to generate the ML datasets.

  

The probes can be generated from this data using their corresponding scripts: 
`python3 -m data.build_probe_<$PROBE_NUMBER>_data` where the the probe number matches:

  

`probe_1`: movie to movie (recommendation probe)

  

`probe_2`: tag to movie (attribute probe)

  

`probe_3`: movie to review (description probe)

  

`probe_4`: movie and tag to movie (combination probe)

  

The probes run on two cutoffs: `probe_min_pop` and `popular_min_pop` which determine the lowest
popularity level of movies included in the probe and movies considered fro the random popular negative.
Popularity is calculated as the number of sequences containing a given movie in the ML 
sequences dataset. The default levels are: 30 for `probe_min_pop` in order to filter out rare movies, 
misspellings, and alternate titles, and 138 for `popular_min_pop` in order to only consider the top 
10% of movies.