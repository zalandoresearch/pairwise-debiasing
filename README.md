This repository accompanies the paper "A General Framework for Pairwise Unbiased Learning to Rank" published in ICTIR 2022.
It contains code used for experiments described in the paper.

## Reproducing the Experiment Results

In order to recreate our experimental results, please follow the steps below.

1. Install [Docker](https://www.docker.com/).
2. Change to the root directory of the project and run `./run.sh`.

The script will start a docker container, perform simulations, and train and evaluate models.
The results will be produced in the form of summary tables saved in subdirectories under `output`.
Each subdirectory corresponds to a particular experimental setting 
(defined by the click model and the maximum number of displayed positions). See the cited paper for more details.
