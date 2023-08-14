#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
import shutil
import gym
import torch

import argparse
import configparser
from policy.lstm_rl import LstmRL
from policy.followDQN1 import Follower1
from policy.followDQN2 import Follower2
from utils.robot import Robot
from utils.follower import Follower
from utils.memory import ReplayMemory
from utils.updateNetwork import Trainer
from utils.learning import Learning
from env.train_crowd_sim import TrainCrowdSim
from utils.Loging import Log
logging = Log(__name__).getlog()

def main():
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument('--env_config',type=str,default = 'env.config')
    args = parser.parse_args()
    make_new_dir = True
    file_dir = 'data/output.log'
    output_dir = 'data/output'
    if os.path.exists(output_dir) or os.path.exists(file_dir):
        key = raw_input('Output directory already exists! Overwrite the folder? (y/n): ')
        if key == 'y':
            shutil.rmtree(output_dir)
            with open(r'data/output.log', 'a+', ) as test:
                test.truncate(0)
        else:
            make_new_dir = False


    if make_new_dir:
        os.makedirs(output_dir)

    log_file = os.path.join('data','output.log')
    rl_weight_file = os.path.join('data', 'rl_model')
    follower1_weight_file = os.path.join('data', 'follower1_model')
    follower2_weight_file = os.path.join('data', 'follower2_model')

    policy = LstmRL()
    policy1 = Follower1()
    policy2 = Follower2()

    #env = gym.make('CrowdSim-v0')
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = TrainCrowdSim()
    env.configure(env_config)

    policy.configure(env_config)
    policy1.configure(env_config)
    policy2.configure(env_config)

    robot = Robot()
    follower1 = Follower()
    follower2 = Follower()

    robot.configure(env_config)
    follower1.configure(env_config)
    follower2.configure(env_config)
    policy.configure(env_config)
    policy1.configure(env_config)
    policy2.configure(env_config)

    robot.set_policy(policy)
    follower1.set_policy(policy1)
    follower2.set_policy(policy2)
    env.set_robot(robot)
    env.set_follower(follower1, follower2)

    rl_learning_rate = 0.0001
    train_batches = 100
    train_episodes = 10001
    target_update_interval = 5
    capacity = 100000
    epsilon_start = 0.2
    epsilon_end = 0.02
    epsilon_decay = 1000
    checkpoint_interval = 100
    episode = 0
    memory_flag = True

    model = robot.policy.get_model()
    model1 = follower1.policy.get_model()
    model2 = follower2.policy.get_model()
    memory = ReplayMemory(capacity)

    batch_size = 100
    updateNetwork = Trainer(model,model1,model2, memory, batch_size)
    learning = Learning(env, robot, memory, model=model, model1=model1, model2=model2, follower1=follower1, follower2=follower2)
    policy.set_env(env)
    policy1.set_env(env)
    policy2.set_env(env)
    robot.set_policy(policy)
    updateNetwork.set_learning_rate(rl_learning_rate)

    if os.path.exists(rl_weight_file):
        model.load_state_dict(torch.load(rl_weight_file))
        logging.info('Load reinforcement learning trained weights.')
    if os.path.exists(follower1_weight_file):
        model1.load_state_dict(torch.load(follower1_weight_file))
        logging.info('Load follower1 reinforcement learning trained weights.')
    if os.path.exists(follower2_weight_file):
        model2.load_state_dict(torch.load(follower2_weight_file))
        logging.info('Load follower2 reinforcement learning trained weights.')

    while episode < train_episodes:
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)
        follower1.policy.set_epsilon(epsilon)
        follower2.policy.set_epsilon(epsilon)

        learning.trainEpisode(Episode=episode)
        if len(memory) > 100:
            if memory_flag:
                print("The replay memory is full !!!")
                memory_flag = False
            updateNetwork.optimize_batch(train_batches)
        episode += 1

        if episode % target_update_interval == 0:
            learning.update_target_model(model, model1, model2)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)
            torch.save(model1.state_dict(), follower1_weight_file)
            torch.save(model2.state_dict(), follower2_weight_file)



if __name__ == "__main__":
    main()
