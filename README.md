# NA<sup>2</sup>Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning

Code for **NA<sup>2</sup>Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning** accepted by ICML 2023. NA<sup>2</sup>Q is implemented in PyTorch and tested on challenging tasks [LBF](https://github.com/semitable/lb-foraging) and [SMAC](https://github.com/oxwhirl/smac) as benchmarks, which is based on [PyMARL](https://github.com/oxwhirl/pymarl). [[paper]](https://proceedings.mlr.press/v202/liu23be.html) [[code]](https://github.com/zichuan-liu/NA2Q)


## Python MARL framework

This PyMARL includes baselines of the following algorithms:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**WQMIX**: Weighted Qmix: Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf)
- [**Qatten**: Qatten: A General Framework for Cooperative Multiagent Reinforcement Learning](https://arxiv.org/abs/2002.03939)
- [**QPLEX**: Qplex: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**MASAC**: Actor-Attention-Critic for Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf)
- [**SHAQ**: SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning](https://arxiv.org/pdf/2105.15013.pdf)
- [**SQDDPG**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**UPDET**: Updet: Universal Multi-Agent Reinforcement Learning via Policy Decoupling with Transformers](https://arxiv.org/abs/2101.08001)
- [**CDS**: Celebrating Diversity in Shared Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2106.02195)

Thanks to all the original authors!

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
@inproceedings{liu2023na2q,
  title = 	 {{NA$^2$Q}: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning},
  author =       {Liu, Zichuan and Zhu, Yuanyang and Chen, Chunlin},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {22539--22558},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v202/liu23be.html},
}
```
In case of any questions, bugs, suggestions or improvements, please feel free to drop me or open an issue.

**If you need the experimental results data from our paper, please contact me at *zichuanliu@smail.nju.edu.cn* and I'll be happy to share them!**
