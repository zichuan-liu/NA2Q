import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env

map_dict = {
    "3m":{"ally_num":3,"enemy_num":3},
    "8m":{"ally_num":8,"enemy_num":8},
    "2s3z":{"ally_num":5,"enemy_num":5},
    "3s5z":{"ally_num":8,"enemy_num":8},
    "5m_vs_6m":{"ally_num":5,"enemy_num":6},
    "8m_vs_9m":{"ally_num":8,"enemy_num":9},
    "10m_vs_11m":{"ally_num":10,"enemy_num":11},
    "3s5z_vs_3s6z":{"ally_num":8,"enemy_num":9},
    "2m_vs_1z":{"ally_num":2,"enemy_num":1},
    "2s_vs_1sc":{"ally_num":2,"enemy_num":1},
    "3s_vs_3z":{"ally_num":3,"enemy_num":3},
    "3s_vs_4z":{"ally_num":3,"enemy_num":4},
    "3s_vs_5z":{"ally_num":3,"enemy_num":5},
    "6h_vs_8z":{"ally_num":6,"enemy_num":8},
    "corridor":{"ally_num":6,"enemy_num":24},
    "2c_vs_64zg":{"ally_num":2,"enemy_num":64},
    "1c3s5z":{"ally_num":9,"enemy_num":9},
    "MMM":{"ally_num":10,"enemy_num":10},
    "MMM2":{"ally_num":10,"enemy_num":12},
    "7sz":{"ally_num":14,"enemy_num":14},
    "5s10z":{"ally_num":15,"enemy_num":15},
    "1c3s5z_vs_1c3s6z":{"ally_num":9,"enemy_num":10},
    "1c3s8z_vs_1c3s9z":{"ally_num":12,"enemy_num":13},
    "pp-2":{"ally_num":8,"enemy_num":8},
    "pp-1":{"ally_num":8,"enemy_num":8},
    "pp-0.5":{"ally_num":8,"enemy_num":8},
    "lbf-4-2":{"ally_num":4,"enemy_num":4},
    "lbf-4-4":{"ally_num":4,"enemy_num":4},
    "lbf-3-3":{"ally_num":3,"enemy_num":3},
}


def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]   

    if map_name not in map_dict:
        print("the map is not in the dict")
        return 
    else:
        args.ally_num = map_dict[map_name]["ally_num"]
        args.enemy_num = map_dict[map_name]["enemy_num"]
        
    alg_name = _config["name"]
    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    args.alg_name = alg_name
    args.map_name = map_name
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs",str(map_name),str(alg_name))
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner, learner):

    for i in range(args.test_nepisode):
        print("test_nepisode: ", i)
        if args.mixer == "qnam":
            runner.run(test_mode=True, vae=learner.eval_diff_network, mixer=learner.mixer)
        else:
            runner.run(test_mode=True, mixer=learner.mixer)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    if 'sc2' in args.env:
        args.unit_dim = runner.env.shield_bits_ally + runner.env.unit_type_bits + 4
        print("move_feat: ", runner.env.get_obs_move_feats_size())
        print("n_enemy, enemy_feat: ", runner.env.get_obs_enemy_feats_size())
        print("n_ally, ally_feat: ", runner.env.get_obs_ally_feats_size())
        print("own_feat: ", runner.env.get_obs_own_feats_size())
    elif 'stag_hunt' in args.env:
        args.unit_dim = runner.env.x_max * runner.env.y_max
    elif 'foraging' in args.env:
        args.unit_dim = 0
    print("args.state_shape: ", args.state_shape)
    print("args.obs_shape: ", args.obs_shape)
    print("args.n_actions: ", args.n_actions)
    print("args.n_agents: ", args.n_agents)
    print("args.unit_dim: ", args.unit_dim)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          burn_in_period=args.burn_in_period,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    total_mac_params = sum(p.numel() for p in list(mac.agent.parameters()))
    total_mix_params = sum(p.numel() for p in list(learner.mixer.parameters()))
    print(total_mac_params+total_mix_params)
    print(total_mac_params+total_mix_params)
    print(total_mac_params+total_mix_params)
    print(total_mac_params+total_mix_params)
    print(total_mac_params+total_mix_params)
    print(total_mac_params+total_mix_params)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner, learner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.alg_name+"_"+args.map_name, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


# TODO: Clean this up
def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    # assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
    #     "need to use replay buffer if running in parallel mode!"

    # assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    # if config["learner"] == "coma":
    #    assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
    #    (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
    #        "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config
