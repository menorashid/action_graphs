## Paper
This repository provides code for:

[Action Graphs: Weakly-supervised Action Localization with Graph Convolution Networks.](https://arxiv.org/abs/1704.04023) Maheen Rashid, Hedvig Kjellström, Yong Jae Lee. WACV 2020.

If you find this repo useful please cite our work:
```bib
@inproceedings{rashid2020action,
  title={Action Graphs: Weakly-supervised Action Localization with Graph Convolution Networks},
  author={Rashid, Maheen and Kjellstr{\"o}m, Hedvig and Lee, Yong Jae},
  booktitle={Winter Conference on Applications of Computer Vision},
  year={2020}
}
```
For questions contact Maheen Rashid (mhnrashid at ucdavis dot edu)

## Getting Started

Download the code from GitHub:
```bash
git clone https://github.com/menorashid/action_graphs.git
cd action_graphs
```

This repo requires Python 2.7. 

It'll be easiest to set up a virtual environment for this repo. 
```bash
pip install virtualenv
virtualenv --python=/usr/bin/python2.6 <path/to/action_graphs_venv>
```

Activate virtual env and install all needed requirements
```bash
source <path/to/action_graphs_venv>/bin/activate
cd <path/to/action_graph_git_repo>
pip install -r requirements.txt
```

## Dataset
Download features extracted from Kinetrics pretrained i3D 
[UCF101 Features](https://www.dropbox.com/s/cjkfpq6n6l0zan4/i3d_features.tar.gz) (5.5 GB)
[ActivityNet Features] TBD
[Charades Features] TBD

Run the following commands
```bash
cd data
tar -xzvf <path to data tar file>
```

Thank you so much to Sujoy Paul for originally sharing the UCF101 and ActivityNet features [here](https://github.com/sujoyp/wtalc-pytorch)

## Models
To download the pretrained models go [here](https://www.dropbox.com/s/eoz0946ifeac1wd/action_graphs.tar.gz) (90 MB)

Run the following commands
```bash
cd experiments
tar -xzvf <path to models tar file>
```
<!-- Otherwise add the individual models to *experiments/*
* [ActionGraphs on UCF101](https://www.dropbox.com/s/g0e7tj2r708eue1/horse_full_model_tps.dat)(36 MB)
* [ActionGraphs on ActivityNet](https://www.dropbox.com/s/3vj7nts5f1v0ry0/horse_full_model_affine.dat)(63 MB)
* [ActionGraphs on Charades](https://www.dropbox.com/s/3un0dild6xar8uf/horse_tps_model.dat)(34 MB)
 -->
## Testing
To test pretrained model run the following line after uncommenting the relevant lines in main
```bash
cd code
python test_pretrained.py
```

## Training
Script for training the full model with varying values of *d* as shown in Table 4 in the paper (uncomment relevant lines in main):
```bash
cd code
python exps_deno.py
```
Script for training the models from Table 2 and Table 3 in the paper (uncomment relevant lines in main):
```bash
cd code
python exps_ablation_etc.py
```

This will save and log training in the following dir
```bash
experiments/<model name>/<model params>/<training params>/
```

The output files would be similar to the pretrained model data shared above and include a .txt log file, .png graphs with losses and accuracy, and .pt model files. 

Please email any questions to me (mhnrashid at ucdavis dot edu). I may not get around to looking at Git issues. Thank you for looking at our paper and code! 