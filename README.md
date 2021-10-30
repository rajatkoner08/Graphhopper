# Graphhopper: Multi-Hop Scene Graph Reasoning for Visual Question Answering


This repository contains data and code (learning agent) for the papers [Graphhopper: Multi-Hop Scene Graph Reasoning for Visual Question Answering(ISWC 2021,Best Paper nomination)](https://arxiv.org/abs/2107.06325) and [Scene Graph Reasoning for Visual Question Answering
(ICML workshop,2020)
](https://arxiv.org/abs/2107.05448). For GQA scene graph has been generated through [Relation Transformer Network](https://github.com/rajatkoner08/rtn) github repo. If you like the paper, please cite our work:

### Bibtex

```
@inproceedings{koner2021graphhopper,
  title={Graphhopper: Multi-hop Scene Graph Reasoning for Visual Question Answering},
  author={Koner, Rajat and Li, Hang and Hildebrandt, Marcel and Das, Deepan and Tresp, Volker and G{\"u}nnemann, Stephan},
  booktitle={International Semantic Web Conference},
  pages={111--127},
  year={2021},
  organization={Springer}
}
@article{hildebrandt2020scene,
  title={Scene graph reasoning for visual question answering},
  author={Hildebrandt, Marcel and Li, Hang and Koner, Rajat and Tresp, Volker and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2007.01072},
  year={2020}
}
```
This directory contains code for the agent - the reasoning submodule of the proposed Graphhopper.


## Setup
This project is being developed with python3.7
```shell
$ python -m pip install -r requirements.txt
$ sh install.sh # this line installs pytorch_geometric
```

```
In case of problem, refer to [here](https://github.com/rusty1s/pytorch_geometric) for more information about pytorch_geometric.
```
```
$ export PYTHONPATH=$PYTHONPATH:/directory/to/VQA/
``` 
## Download Data
(Required) Download the word embeddings [glove.6B.200d.pth](https://drive.google.com/file/d/1ptU4oHObYoqm1se5ZiBQrX_5IZsMFsWm/view?usp=sharing) 
into the directory of "VQA/data/glove.6B.200d.pth"


## Process Data
Data structure for the agent:

    |- root e.g. scenegraphs
        |- graphs: folder of idx.json graphs
        |- *.json: dictionary files
        |- *.txt: question files

Build the NetworkxGraphs for the RL agent from SG Predictor (pkl files, you can find a sample file at VQA/data/dicts/sample.pkl):

1. Build the graphs. This will output the NetworkxGraphs in the directory of "scenegraphs/graphs":
```shell
$ cd codes/preprocessing
$ python build_graph.py PATH_TO_TRAIN_PKL PATH_TO_VAL_PKL```
```
2. Create some dictionaries. This will output some .json files in the directory of "scenegraphs"
```
$ cd scenegraphs
$ python ../get_all.py
```

## Train
```
$ cd VQA
```
For debug:
```
$ python codes/trainer.py configs/test_toy_dataset.yaml
```
For real training:
```
$ python codes/trainer.py configs/query.yaml
or
$ python codes/trainer.py configs/binary.yaml
```

#Help
Please feel free to open an issue if you encounter trouble getting it to work.