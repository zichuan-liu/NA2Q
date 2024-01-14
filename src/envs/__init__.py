from functools import partial
from smac.env import MultiAgentEnv
from .starcraft import StarCraft2Env
import sys
import os
from .stag_hunt import StagHunt
from .foraging import ForagingEnv
from .mpe.mpe_wrapper import MPEWrapper

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)
def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["foraging"] = partial(env_fn, env=ForagingEnv)
REGISTRY["mpe"] = partial(env_fn, env=MPEWrapper)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)
if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
