# Efficient Graph Neural Network Inference at Large Scale

## Introduction
This repository is the implementation of Node-Adaptive Inference (NAI). 


## Requirements
All experiments are implemented in Python 3.9 with Pytorch 1.11.0.
To install requirements:

```setup
pip install -r requirements.txt
```

## Data

The datasets tested in our experiments, including PubMed, Flickr, Ogbn-axiv, Ogbn-products are downloaded from Deep Graph Library ([DGL](https://docs.dgl.ai/api/python/dgl.data.html)) and Open Graph Benchmark ([OGB](https://ogb.stanford.edu/docs/nodeprop/)).

## Training and inference

To train the base model and execute the Inception distillation:

```bash
$ cd scr
$ bash run.sh
```

## Parameter selection for inference

To search the parameters for Node-Adaptive Propagation (NAP) and select the appropriate inference pattern, run `./scr/para_search.py` and `./scr/para_selection.py`, respectively.

The folder `./scr/results/` contains the trained models, searched parameters and inference results on Flickr dataset.
