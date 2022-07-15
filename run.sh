#!/bin/sh

IMAGE="python:3.8.13-bullseye"
WORKING_DIR="/home/pairwise-debiasing"

docker pull $IMAGE
docker run -it -v "$(pwd)":$WORKING_DIR -w $WORKING_DIR $IMAGE /bin/bash experiment.sh