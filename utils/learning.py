import logging
import copy
import torch
from utils.info import *
from math import pow, pi, sqrt, atan2, cos, sin

from utils.Loging import Log
logging = Log(__name__).getlog()



class Learning(object):
    def __init__(self, env, robot, memory=None, model=None, model1=None, model2=None, follower1=None, follower2=None):
        self.env = env
        self.robot = robot
        self.follower1 = follower1
        self.follower2 = follower2
        self.memory = memory
        self.target_model = model
        self.target_model1 = model1
        self.target_model2 = model2

    def update_target_model(self, target_model, target_model1, target_model2):
        self.target_model = copy.deepcopy(target_model)
        self.target_model1 = copy.deepcopy(target_model1)
        self.target_model2 = copy.deepcopy(target_model2)

    # @profile
    def trainEpisode(self, Episode=None):
        success = 0
        collision = 0
        timeout = 0

        ob = self.env.reset()
        done = False
        states, states1, states2 = [], [], []
        rewards, rewards1, rewards2 = [], [], []
        flag = 0
        sum_pow_d = 0
        sum_length = 0
        vp = []
        info = None

        while not done:
            if sum_length < 12:
                rg_dist = sqrt(pow(self.robot.px - self.robot.gx, 2) + pow(self.robot.py - self.robot.gy, 2))
                if rg_dist > 1.5:
                    sum_length += 0.5 * self.env.time_step
                    xr = sum_length
                    yr = 5 * sin((sum_length * pi) / 6)
                    vp.append(0.18)
                    self.env.set_robot_goal_position(xr, yr)
                else:
                    sum_length += 1.0 * self.env.time_step
                    xr = sum_length
                    yr = 5 * sin((sum_length * pi) / 6)
                    vp.append(1.4)
                    self.env.set_robot_goal_position(xr, yr)
            else:
                sum_length += 0.01
                xr = sum_length
                yr = 5 * sin((sum_length * pi) / 6)
                self.env.set_robot_goal_position(xr, yr)
            action0 = self.robot.act(ob)
            exclusiveState1 = [self.follower1.px, self.follower1.py, self.follower2.px, self.follower2.py]
            action1 = self.follower1.act(ob, exclusiveState1)
            exclusiveState2 = [self.follower1.px + action1.vx * self.env.time_step, self.follower1.py + action1.vy * self.env.time_step, self.follower2.px, self.follower2.py]
            action2 = self.follower2.act(ob, exclusiveState2)

            action = [action0, action1, action2]
            ob, reward, done, info, reward1, reward2 = self.env.step(action)
            flag += 1
            sum_pow_d += pow(self.robot.px - self.robot.gx, 2) + pow(self.robot.py - self.robot.gy, 2)

            states.append(self.robot.policy.last_state)
            states1.append(self.follower1.policy.last_state)
            states2.append(self.follower2.policy.last_state)
            rewards.append(reward)
            rewards1.append(reward1)
            rewards2.append(reward2)

        if sum_length >= 12 and isinstance(info, ReachGoal):
            success += 1
            nav_time = self.env.global_time
        elif isinstance(info, Collision):
            collision += 1
            nav_time = self.env.global_time
        elif isinstance(info, Timeout):
            timeout += 1
            nav_time = self.env.global_time
        else:
            raise ValueError('Invalid end signal from environment')

        if isinstance(info, ReachGoal) or isinstance(info, Collision):
            self.updateMemory(states, rewards, states1, rewards1, states2, rewards2)

        cumulative_rewards = 0.02*sum(rewards)

        v_mean_loss = 0.9 * sqrt(sum_pow_d / flag)

        extra_info = '' if Episode is None else 'in Episode {} '.format(Episode)
        logging.info('{}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},V_MeanSquareError:{:.4f},times:{:d}'.
            format(extra_info, success, collision, nav_time, cumulative_rewards, v_mean_loss, flag))
        if success:
            logging.info('Has succes Sum_MeanSquareError:{:.4f}, nav_time: {:.2f}'.format(v_mean_loss, nav_time))

    def updateMemory(self, states, rewards, states1, rewards1, states2, rewards2):
        if self.memory is None:
            raise ValueError('Memory value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]
            reward1 = rewards1[i]
            reward2 = rewards2[i]
            state1 = states1[i]
            state2 = states2[i]
            if i == len(states) - 1:
                value = 0.1 * reward
                value1 = 0.1 * reward1
                value2 = 0.1 * reward2
            else:
                next_state = states[i + 1]
                value = 0.1 * reward + 0.9 * self.target_model(next_state.unsqueeze(0)).data.item()

                next_state1 = states1[i + 1]
                value1 = 0.1 * reward1 + 0.9 * self.target_model1(next_state1).data.item()

                next_state2 = states2[i + 1]
                value2 = 0.1 * reward2 + 0.9 * self.target_model2(next_state2).data.item()

            self.memory.push((state, torch.Tensor([value]), state1, torch.Tensor([value1]), state2, torch.Tensor([value2])))
