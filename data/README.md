# Data for the E2E Convrec Project

`data/redial` contains the redial dataset as unpacked from [the redial website](https://redialdata.github.io/website/) (`rd-test-data.jsonl`, `rd-train-data.jsonl`) as well as the vesion that has been reformatted for training (`rd-train-formatted.jsonl`, `rd-test-formatted.jsonl`). 

`build_redial.py` contains the script to generate the formatted data from the dataset. There should be no need to rerun the `build_redial.py` script unless it is necessary to rebuild the data with a different format.

you can run the `build_redial.py` script using `python3 -m data.build_redial` from the project root folder or `python3 -m data.build_redial --data_dir=./redial` from this folder