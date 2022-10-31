#!/bin/bash
python 1train_initial.py --dataset flickr --gpu 4
python 2off_distill.py --dataset flickr --gpu 4
python 3on_distill.py --dataset flickr --gpu 4
python 4inference.py --dataset flickr --gpu -1