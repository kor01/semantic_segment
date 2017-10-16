#! /bin/bash

export PYTHONPATH=$(readlink -f ../)
nohup jupyter-notebook > /dev/null &
