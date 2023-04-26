from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from modules.intrinsic.qnam_context import VAE
import torch as th
import os
class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, vae=None, mixer=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            if self.args.evaluate and test_mode and vae:
                actions, hidden, agent_outputs = self.mac.select_actions_vis(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                mask, _, _ = vae(hidden)
                latent, _, _ = vae.encode(hidden)
                inp_state = self.env.get_state()
                inp_state = th.from_numpy(inp_state).to(self.args.device)
                # Pick the Q-Values for the actions taken by each agent
                chosen_action_qvals = agent_outputs.max(dim=2)[0]
                chosen_action_qvals = chosen_action_qvals.unsqueeze(1)
                q_tot, f_i, w, v = mixer(chosen_action_qvals, inp_state, latent, test=True)
                f_i = f_i.squeeze()
                w = w.squeeze()
                v = v.squeeze()
                af_i = f_i*w
                if 'foraging' in self.args.env:
                    reward, terminated, env_info = self.env.step_with_vis(actions[0], mask, agent_outputs, af_i, v)
                else:
                    self.save_credit(q_tot, chosen_action_qvals, self.t, self.env.get_obs(), actions, f_q_i=af_i, w=w, imp_fearture=mask)
                    reward, terminated, env_info = self.env.step(actions[0])
            elif self.args.evaluate and test_mode:
                actions, hidden, agent_outputs = self.mac.select_actions_vis(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                inp_state = self.env.get_state()
                inp_state = th.from_numpy(inp_state).to(self.args.device)
                chosen_action_qvals = agent_outputs.max(dim=2)[0]
                chosen_action_qvals = chosen_action_qvals.unsqueeze(1)
                q_tot = mixer(chosen_action_qvals, inp_state).squeeze(0)
                chosen_action_qvals = chosen_action_qvals.squeeze(0)
                # save credit to analysis
                self.save_credit(q_tot,chosen_action_qvals, self.t, self.env.get_obs(), actions)
                reward, terminated, env_info = self.env.step(actions[0])
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        if 'foraging' in self.args.env:
            env_info = {}
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def save_credit(self, q_tot, q_i, t, obs, actions, f_q_i=None, imp_fearture=None, w=None, credit=None):
        save_path = os.path.join(self.args.local_results_path, "credit", self.args.map_name, self.args.alg_name+str(self.args.seed))
        os.makedirs(save_path, exist_ok=True)
        filename = save_path+'/q_tot.npy'
        if os.path.exists(filename):
            pre_data = np.load(filename)
            q_tot = np.concatenate([pre_data,q_tot], 0)
        np.save(filename, q_tot)

        filename = save_path+'/q_i.npy'
        if os.path.exists(filename):
            pre_data = np.load(filename)
            q_i = np.concatenate([pre_data, q_i], 0)
        np.save(filename, q_i)

        filename = save_path+'/obs.npy'
        obs = np.array(obs)
        obs = obs.reshape(1, obs.shape[0], obs.shape[1])
        if os.path.exists(filename):
            pre_data = np.load(filename)
            obs = np.concatenate([pre_data, obs], 0)
        np.save(filename, obs)

        filename = save_path+'/t.npy'
        t = np.array([t])
        if os.path.exists(filename):
            pre_data = np.load(filename)
            t = np.concatenate([pre_data, t], 0)
        np.save(filename, t)

        filename = save_path+'/actions.npy'
        if os.path.exists(filename):
            pre_data = np.load(filename)
            actions = np.concatenate([pre_data, actions], 0)
        np.save(filename, actions)

        if f_q_i!=None:
            f_q_i = f_q_i.reshape(1, -1)
            filename = save_path+'/f_q_i.npy'
            if os.path.exists(filename):
                pre_data = np.load(filename)
                f_q_i = np.concatenate([pre_data,f_q_i], 0)
            np.save(filename, f_q_i)

        if imp_fearture!=None:
            imp_fearture = imp_fearture.reshape(1, imp_fearture.shape[0], imp_fearture.shape[1])
            filename = save_path+'/imp_fearture.npy'
            if os.path.exists(filename):
                pre_data = np.load(filename)
                imp_fearture = np.concatenate([pre_data,imp_fearture], 0)
            np.save(filename, imp_fearture)

        if credit!=None:
            credit = credit.reshape(1, -1)
            filename = save_path+'/credit.npy'
            if os.path.exists(filename):
                pre_data = np.load(filename)
                credit = np.concatenate([pre_data, credit], 0)
            np.save(filename, credit)

        if w!=None:
            w = w.reshape(1, -1)
            filename = save_path+'/w.npy'
            if os.path.exists(filename):
                pre_data = np.load(filename)
                w = np.concatenate([pre_data, w], 0)
            np.save(filename, w)