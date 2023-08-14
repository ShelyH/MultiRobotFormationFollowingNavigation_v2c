import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from utils.learning import Learning
from utils.robot import Robot
from utils.follower import Follower
# from policy.orca import ORCA
from policy.lstm_rl import LstmRL
from policy.followDQN1 import Follower1
from policy.followDQN2 import Follower2
import random

from env.train_crowd_sim import TrainCrowdSim
from math import pow, pi, sqrt, atan2, cos, sin
from utils.info import *
from utils.Loging import Log

logging = Log(__name__).getlog()


def main():
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument('--env_config', type=str, default='env.config')
    args = parser.parse_args()
    model_weights = os.path.join('data', 'rl_model')
    model_weights1 = os.path.join('data', 'follower1_model')
    model_weights2 = os.path.join('data', 'follower2_model')
    print('use rl_model')

    policy = LstmRL()
    policy1 = Follower1()
    policy2 = Follower2()

    policy.get_model().load_state_dict(torch.load(model_weights))
    policy1.get_model().load_state_dict(torch.load(model_weights1))
    policy2.get_model().load_state_dict(torch.load(model_weights2))

    # configure environment
    # env = gym.make('CrowdSim-v1')
    env = TrainCrowdSim()

    env.run_env = 'test'
    policy.phase = 'test'
    policy1.phase = 'test'
    policy2.phase = 'test'
    robot = Robot()
    follower1 = Follower()
    follower2 = Follower()
    robot.set_policy(policy)
    follower1.set_policy(policy1)
    follower2.set_policy(policy2)

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env.configure(env_config)

    robot.configure(env_config)
    follower1.configure(env_config)
    follower2.configure(env_config)
    policy.configure(env_config)
    policy1.configure(env_config)
    policy2.configure(env_config)

    print(follower2.time_step)

    env.set_robot(robot)
    env.set_follower(follower1, follower2)
    learning = Learning(env, robot, follower1=follower1, follower2=follower2)

    policy.set_phase('test')
    policy.set_env(env)
    policy1.set_phase('test')
    policy1.set_env(env)
    policy2.set_phase('test')
    policy2.set_env(env)

    test_Once = True
    if test_Once:
        print('is visualize')
        ob = env.reset()
        done = False
        last_pos = np.array(robot.get_position())
        flag = 0
        vp = []
        sum_pow_d = 0
        sum_length = 0
        while not done:
            if sum_length < 12:
                rg_dist = sqrt(pow(robot.px - robot.gx, 2) + pow(robot.py - robot.gy, 2))
                if rg_dist > 1.5:
                    sum_length += 0.5 * env.time_step
                    xr = sum_length
                    yr = 5 * sin((sum_length * pi) / 6)
                    vp.append(0.18)
                    env.set_robot_goal_position(xr, yr)
                else:
                    sum_length += 1.0 * env.time_step
                    xr = sum_length
                    yr = 5 * sin((sum_length * pi) / 6)
                    vp.append(1.4)
                    env.set_robot_goal_position(xr, yr)
            else:
                sum_length += 0.01
                xr = sum_length
                yr = 5 * sin((sum_length * pi) / 6)
                env.set_robot_goal_position(xr, yr)
            action0 = robot.act(ob)
            exclusiveState1 = [follower1.px, follower1.py, follower2.px, follower2.py]
            action1 = follower1.act(ob, exclusiveState1)
            exclusiveState2 = [follower1.px + action1.vx * env.time_step, follower1.py + action1.vy * env.time_step,
                               follower2.px, follower2.py]
            action2 = follower2.act(ob, exclusiveState2)

            action = [action0, action1, action2]
            ob, reward, done, info, reward1, reward2 = env.step(action)
            current_pos = np.array(robot.get_position())

            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
            flag += 1
            sum_pow_d += pow(robot.px - robot.gx, 2) + pow(robot.py - robot.gy, 2)

        average_V = env.sum_V / env.MSE_Flag
        for i in range(len(env.V_list)):
            env.MSE_V += pow(env.V_list[i] - average_V, 2)

        MSE_Vel = sqrt(env.MSE_V / env.MSE_Flag)
        logging.info('Mean square error of velocity is %.2f', MSE_Vel)
        print('MSE_Theta', env.MSE_Theta)
        MSE_The = sqrt(env.MSE_Theta / (env.MSE_Flag - 1))
        logging.info('Mean square error of angle is %.2f', MSE_The)
        print('robot Velocity:', env.V_list)
        print("path velocity:", vp)
        print("angle list:", env.theta_list)
        P_MSE = sqrt(sum_pow_d / flag)
        print("MSE_P:", P_MSE)
        env.render()
        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
    else:
        print('no visualize')
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
