import torch
from numpy.lib.ufunclike import isposinf
from smac.env.multiagentenv import MultiAgentEnv
import numpy as np
import gym
from gym.envs.registration import register
from sys import stderr
import pygame
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import lbforaging
from matplotlib import gridspec
class ForagingEnv(MultiAgentEnv):

    def __init__(self,
                 field_size: int,
                 players: int,
                 max_food: int,
                 force_coop: bool,
                 partially_observe: bool,
                 sight: int,
                 is_print: bool,
                 seed: int,
                 need_render: bool,
                 render_output_path: str = '',
                 **kwargs):
        self.n_agents = players
        self.n_actions = 6
        self._total_steps = 0
        self._episode_steps = 0
        self.is_print = is_print
        self.need_render = need_render
        np.random.seed(seed)

        self.episode_limit = 50

        self.agent_score = np.zeros(players)

        register(
            id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v0".format(field_size, players, max_food,
                                                            "-coop" if force_coop else "",
                                                            "-{}s".format(sight) if partially_observe else ""),
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": players,
                "max_player_level": 3,
                "field_size": (field_size, field_size),
                "max_food": max_food,
                "sight": sight if partially_observe else field_size,
                "max_episode_steps": 50,
                "force_coop": force_coop,
            },
        )

        env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v0".format(field_size, players, max_food,
                                                              "-coop" if force_coop else "",
                                                              "-{}s".format(sight) if partially_observe else "")
        print('Env:', env_id, file=stderr)
        print('Env:', env_id, file=stderr)
        print('Env:', env_id, file=stderr)
        self.env = gym.make(env_id)
        self.env.seed(seed)

        if self.need_render:
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            render_path = os.path.join(render_output_path, 'lbf_vis', date)
            if not os.path.exists(render_path):
                os.makedirs(render_path, exist_ok=True)
            self.render_path = render_path

            # Enable interactive mode.
            plt.ion()
            # Create a figure and a set of subplots.
            self.figure, self.ax = plt.subplots(figsize=(6,4), dpi=600)
            self.figure.clf()
            gs = gridspec.GridSpec(4, 5, wspace=-0.67, hspace=0.4)
            self.ax1 = self.figure.add_subplot(gs[:, :4])
            self.ax2 = self.figure.add_subplot(gs[0, 4])
            self.ax3 = self.figure.add_subplot(gs[1, 4])
            self.ax4 = self.figure.add_subplot(gs[2, 4])
            self.ax5 = self.figure.add_subplot(gs[3, 4])
            self.ax1.set_xlim(0, 501)
            self.ax1.set_ylim(0, 501)
            # self.ax1.axis('off')
            # self.ax2.axis('off')
            # self.ax3.axis('off')
            # self.ax4.axis('off')
            # self.ax5.axis('off')

    def step(self, actions):
        """ Returns reward, terminated, info """
        self._total_steps += 1
        self._episode_steps += 1

        if self.is_print:
            print(f'env step {self._episode_steps}', file=stderr)
            print('t_steps: %d' % self._episode_steps, file=stderr)
            print('current position: ', file=stderr)
            print(self.env.unwrapped.get_players_position(), file=stderr)
            print('choose actions: ', file=stderr)
            print(actions.cpu().numpy().tolist(), file=stderr)
        position_record = self.env.unwrapped.get_players_position()
        action_record = actions.cpu().numpy().tolist()
        env_info = {
            'agent_position': position_record,
            'agent_action': action_record
        }
            # import pickle
            # pickle.dump(env_info, open(os.path.join(self.render_path, f'info_step_{self._episode_steps}.pkl'), 'wb'))

        if self.need_render:
            fig = plt.figure(dpi=400)
            data = self.env.render(mode='rgb_array')
            plt.imshow(data)
            plt.axis('off')
            fig.savefig(os.path.join(self.render_path, f'image_step_{self._total_steps}.png'), bbox_inches='tight',
                        dpi=600)

        self.obs, rewards, dones, info = self.env.step(actions.cpu().numpy())
        self.agent_score += rewards

        reward = np.sum(rewards)
        # step penalty
        reward -= 0.002
        terminated = np.all(dones)

        return reward, terminated, env_info

    def step_with_vis(self, actions, mask=None, q_values=None, af_i=None, bias=None):
        """ Returns reward, terminated, info """
        self._total_steps += 1
        self._episode_steps += 1

        if self.is_print:
            print(f'env step {self._episode_steps}', file=stderr)
            print('t_steps: %d' % self._episode_steps, file=stderr)
            print('current position: ', file=stderr)
            print(self.env.unwrapped.get_players_position(), file=stderr)
            print('choose actions: ', file=stderr)
            print(actions.cpu().numpy().tolist(), file=stderr)
        position_record = self.env.unwrapped.get_players_position()
        action_record = actions.cpu().numpy().tolist()
        env_info = {
            'agent_position': position_record,
            'agent_action': action_record
        }
            # import pickle
            # pickle.dump(env_info, open(os.path.join(self.render_path, f'info_step_{self._episode_steps}.pkl'), 'wb'))

        agent_q = []
        q_values = q_values.squeeze(0)
        for i in range(len(actions)):
            agent_q.append(q_values[i, actions[i]])
        agent_q = np.array(agent_q)
        print("agent_q: ", agent_q)
        curr_obs = self.get_obs()

        if self.need_render:
            data = self.env.render(mode='rgb_array')
            data[data>250]=5
            data[data<2]=250
            data[500, :, :]=5
            data[:, 0, :]=5
            self.ax1.imshow(data[:, :, :], extent=(0, 501, 0, 501), origin='upper')

            print(position_record, data.shape)
            for i, player_pos in enumerate(position_record):
                mask_arr = np.zeros((250, 250))
                my_obs = curr_obs[i]
                for k in range(0, len(my_obs), 3):
                    if my_obs[k+2]!=0:
                        print(my_obs[k], my_obs[k+1])
                        if player_pos[0]==0:
                            my_obs[k] +=2
                        elif player_pos[0]==1:
                            my_obs[k] +=1
                        if player_pos[1]==0:
                            my_obs[k+1] +=2
                        elif player_pos[1]==1:
                            my_obs[k+1] +=1

                        print(int(my_obs[k])+player_pos[0]-2, int(my_obs[k+1])+player_pos[1]-2, player_pos)
                        mask_arr[(4-int(my_obs[k]))*50:(5-int(my_obs[k]))*50, (int(my_obs[k+1]))*50:int(my_obs[k+1]+1)*50] = mask[i][k+2]
                        print(mask_arr.shape, mask[i][k+2])

                # q_value
                # print("获取系统中所有可用字体", pygame.font.get_fonts())
                # font = pygame.font.Font("/usr/share/fonts/Lato-Light.ttf", 16)
                # rtext = font.render('%.3f'%agent_q[i], True, (100, 100, 100), (100, 100, 100))
                # self.ax.imshow(rtext, alpha=0.5, extent=(player_pos[1]*50+10, player_pos[1]*50+40, (10-player_pos[0])*50-30, (10-player_pos[0])*50-10), cmap='rainbow', origin='upper', vmin=0, vmax=1)
                # plt.text(player_pos[1]*50+10, (10-player_pos[0])*50-30, round(agent_q[i], 3), fontsize=4, bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
                # self.ax.annotate('%.3f'%agent_q[i], xy=(player_pos[1]*50+10, (10-player_pos[0])*50-30), xytext=(player_pos[1]*50+20, (10-player_pos[0])*50-10), arrowprops=dict(facecolor='black', shrink=0.05))
                # FI
                self.ax1.imshow(mask_arr, alpha=0.35, extent=((player_pos[1]-2)*50, (player_pos[1]+3)*50, (12-player_pos[0])*50, (7-player_pos[0])*50), cmap='coolwarm', origin='upper', vmin=0, vmax=mask.max())
                data[player_pos[0], player_pos[1]] = -1

                if i==0:
                    self.ax2.imshow(mask_arr, origin='lower', vmin=0, vmax=mask.max())
                    self.ax2.set_ylabel("pos:({}, {})".format(player_pos[0]+1, player_pos[1]+1), fontsize=5, labelpad=-0.5)
                    self.ax2.set_title(r"$Q_{{{}}}:{{{}}} \rightarrow \alpha_{} f_{{{}}}(Q_{{{}}}):{{{}}}$".format(i+1, '%.2f'%agent_q[i], i+1, i+1, i+1, '%.2f'%af_i[i]), fontsize=5, pad=2.3)
                    self.ax2.set_yticks([])
                    self.ax2.set_xticks([])
                elif i==1:
                    self.ax3.imshow(mask_arr, origin='lower', vmin=0, vmax=mask.max())
                    self.ax3.set_ylabel("pos:({}, {})".format(player_pos[0]+1, player_pos[1]+1), fontsize=5, labelpad=-0.5)
                    self.ax3.set_title(r"$Q_{{{}}}:{{{}}} \rightarrow \alpha_{} f_{{{}}}(Q_{{{}}}):{{{}}}$".format(i+1, '%.2f'%agent_q[i], i+1, i+1, i+1, '%.2f'%af_i[i]), fontsize=5, pad=2.3)
                    self.ax3.set_yticks([])
                    self.ax3.set_xticks([])
                elif i==2:
                    self.ax4.imshow(mask_arr, origin='lower', vmin=0, vmax=mask.max())
                    self.ax4.set_ylabel("pos:({}, {})".format(player_pos[0]+1, player_pos[1]+1), fontsize=5, labelpad=-0.5)
                    self.ax4.set_title(r"$Q_{{{}}}:{{{}}} \rightarrow \alpha_{} f_{{{}}}(Q_{{{}}}):{{{}}}$".format(i+1, '%.2f'%agent_q[i], i+1, i+1, i+1, '%.2f'%af_i[i]), fontsize=5, pad=2.3)
                    self.ax4.set_yticks([])
                    self.ax4.set_xticks([])
                elif i==3:
                    self.ax5.imshow(mask_arr, origin='lower', vmin=0, vmax=mask.max())
                    self.ax5.set_ylabel("pos:({}, {})".format(player_pos[0]+1, player_pos[1]+1), fontsize=5, labelpad=-0.5)
                    self.ax5.set_title(r"$Q_{{{}}}:{{{}}} \rightarrow \alpha_{} f_{{{}}}(Q_{{{}}}):{{{}}}$".format(i+1, '%.2f'%agent_q[i], i+1, i+1, i+1, '%.2f'%af_i[i]), fontsize=5, pad=2.3)
                    self.ax5.set_yticks([])
                    self.ax5.set_xticks([])
            # plt.axis('off')
            self.ax1.set_xticks((range(25, 501, 50)), ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"), fontsize=6)
            self.ax1.set_yticks((range(25, 501, 50)), ("10", "9", "8", "7", "6", "5", "4", "3", "2", "1"), fontsize=6)

            self.ax1.set_title(r"$Q_{{tot}} = <{}> + <{}+ {}+ {}+ {}> + <{}+ \cdots +{}>$".format('%.2f'%bias, '%.2f'%af_i[0], '%.2f'%af_i[1],
                                                                          '%.2f' % af_i[2], '%.2f' % af_i[3],
                                                                          '%.2f' % af_i[4], '%.2f' % af_i[-1],
                                                                          ), fontsize=5)
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            # self.ax.clear()
            self.figure.savefig(os.path.join(self.render_path, f'image_step_{self._total_steps}.pdf'), bbox_inches='tight')

        self.obs, rewards, dones, info = self.env.step(actions.cpu().numpy())
        self.agent_score += rewards

        reward = np.sum(rewards)
        # step penalty
        reward -= 0.002
        terminated = np.all(dones)

        return reward, terminated, env_info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return np.array(self.obs[agent_id])

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env.get_observation_space().shape[0]

    def get_state(self):
        state = self.obs[0]
        for i in range(self.n_agents - 1):
            state = np.concatenate([state, self.obs[i + 1]])
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        res = [0] * self.n_actions
        t = self.env.valid_actions[self.env.players[agent_id]]
        for i in range(len(t)):
            res[t[i].value] = 1
        return res

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self._episode_steps = 0
        self.agent_score = np.zeros(self.n_agents)
        self.obs = self.env.reset()
        return self.get_obs(), self.get_state()

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "agent_score": self.agent_score,
        }
        return stats