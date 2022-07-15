#!/bin/sh

# downloading the ranked dataset
DATASET_FILE_NAME="ranked_yahoo_c14_dataset"
pip3 install gdown
gdown 1G1dGCwXD8IvrgHFDAYn0aZdgiLqEM7vZ -O $DATASET_FILE_NAME.zip
unzip $DATASET_FILE_NAME.zip && rm $DATASET_FILE_NAME.zip
mv generate_dataset $DATASET_FILE_NAME

# installing java
apt-get update
apt install -y openjdk-11-jdk openjdk-11-jre
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/

# installing requirements
pip3 install -r requirements.txt

# preprocessing the downloaded dataset to make it loadable by svmlight utils
python3 src/preprocessing.py

# building the cython extension
cd dlambdamart/dlambdamart || exit
../build.sh
cd ../..

# running experiments
python3 main.py
