# NA^2Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning

## Introduction

NA2Q is an **Interpretable Multi-Agent Reinforcement Learning** framework based on [PyMARL](https://github.com/oxwhirl/pymarl) via **Neural Attention Additive Models**, with a paper named "*NA2Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning*". NA2Q is written in PyTorch and tested on challenging tasks [LBF](https://github.com/semitable/lb-foraging) and [SMAC](https://github.com/oxwhirl/smac) as benchmarks.

## Installation instructions

Build the Dockerfile using 

```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
# demo sc2
python src/main.py --config=qnam --env-config=sc2 with env_args.map_name=MMM2 gpu_id=0 t_max=2010000 epsilon_anneal_time=50000 seed=1
# demo foraging
nohup python src/main.py --config=qnam --env-config=foraging with env_args.map_name=lbf-4-2 use_cuda=False seed=1 &
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

## Citing NA2Q

If you use NA2Q in your research, please cite it in BibTeX format:

```tex
@article{liu2023NA2Q,
  title = {NA2Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning},
  author = {Liu, Zichuan and Zhu, Yuanyang and Chen, Chunlin},
  journal = {CoRR},
  volume = {abs/xxxx.xxxx},
  year = {2023},
}
```
