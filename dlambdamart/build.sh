#! /bin/bash

make libargsort.a
python3 ../setup.py build_ext --inplace
make clean