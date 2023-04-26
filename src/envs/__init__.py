from functools import partial
from smac.env import MultiAgentEnv
from .starcraft import StarCraft2Env
import sys
import os
from .stag_hunt import StagHunt
from .foraging import ForagingEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["foraging"] = partial(env_fn, env=ForagingEnv)
if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
