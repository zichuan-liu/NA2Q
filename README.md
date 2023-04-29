# NA<sup>2</sup>Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning

Code for **NA<sup>2</sup>Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning** accepted by ICML 2023. NA<sup>2</sup>Q is implemented in PyTorch and tested on challenging tasks [LBF](https://github.com/semitable/lb-foraging) and [SMAC](https://github.com/oxwhirl/smac) as benchmarks, which is based on [PyMARL](https://github.com/oxwhirl/pymarl). [[paper]](https://arxiv.org/abs/2304.13383) [[code]](https://github.com/zichuan-liu/NA2Q)

## Installation instructions
### Build the Dockerfile using 

```shell
cd docker
bash build.sh
```

### Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.
### Requirements
- Python 3.6+
- pip packages listed in requirements.txt

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recommended).

## Run an experiment 

```shell
# demo sc2
python src/main.py --config=qnam --env-config=sc2 with env_args.map_name=8m_vs_9m gpu_id=0 t_max=2010000 epsilon_anneal_time=50000 seed=1
# demo foraging
nohup python src/main.py --config=qnam --env-config=foraging with env_args.map_name=lbf-4-2 use_cuda=False seed=1 &
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

## Citing NA<sup>2</sup>Q

If you find this repository useful for your research, please cite it in BibTeX format:

```tex
@article{liu2023na2q,
  title={{NA$^2$Q}: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning},
  author = {Liu, Zichuan and Zhu, Yuanyang and Chen, Chunlin},
  journal = {CoRR},
  volume = {abs/2304.13383},
  year = {2023},
}
```
In case of any questions, bugs, suggestions or improvements, please feel free to drop me at yuanyang@smail.nju.edu.cn or open an issue.
