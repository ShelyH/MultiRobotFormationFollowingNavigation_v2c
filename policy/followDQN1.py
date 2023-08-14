import numpy as np

from policy.policy import Policy
from utils.action import ActionXY
import torch
import torch.nn as nn
import itertools
# from policy import Policy


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork1(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super(ValueNetwork1, self).__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value


class Follower1(Policy):
    def __init__(self):
        super(Follower1, self).__init__()
        self.self_state_dim = 10
        self.human_state_dim = 7
        self.joint_state_dim = 17
        self.action_space = None
        self.epsilon = 0.2
        self.env = None
        self.time_step = 0.25
        self.phase = 'train'
        self.model = ValueNetwork1(6 * self.joint_state_dim, [200, 100, 50, 1])
        print('follower1:', self.model)

    def configure(self, config):
        self.time_step = config.getfloat('env', 'time_step')

    def set_env(self, env):
        self.env = env

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self):
        speeds = [0.2, 0.5, 0.8, 1.2, 1.5]
        rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        action_space = [ActionXY(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            action_space.append(ActionXY(np.float(speed * np.cos(rotation)), np.float(speed * np.sin(rotation))))
        self.action_space = action_space
        print("follower1 action space:", len(self.action_space))

    def follower1_step(self, exclusiveState, action):
        exclusiveState[0] += action.vx * self.time_step
        exclusiveState[1] += action.vy * self.time_step
        return exclusiveState

    def predict(self, state, exclusiveState):
        max_action = None
        max_state = None
        probability = np.random.random()
        if self.action_space is None:
            self.build_action_space()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
            next_exclusiveState = self.follower1_step(exclusiveState, max_action)
            ob, reward, done, info = self.env.onestep_lookahead1(max_action)
            batch_next_states = torch.cat(
                [torch.Tensor([state.self_state + next_human_state]) for next_human_state in ob], dim=0)
            extendState = torch.repeat_interleave(torch.Tensor([next_exclusiveState]), 6, dim=0)
            last_state = torch.cat([extendState, self.rotate(batch_next_states)], dim=1)
            last_state = last_state.reshape(1, -1)
            self.last_state = last_state.squeeze()
        else:
            max_min_value = float('-inf')
            for action in self.action_space:
                next_exclusiveState = self.follower1_step(exclusiveState, action)
                ob, reward, done, info = self.env.onestep_lookahead1(action)
                batch_next_states = torch.cat(
                    [torch.Tensor([state.self_state + next_human_state]) for next_human_state in ob], dim=0)
                extendState = torch.repeat_interleave(torch.Tensor([next_exclusiveState]), 6, dim=0)
                current_state = torch.cat([extendState, self.rotate(batch_next_states)], dim=1)
                current_state = current_state.reshape(1, -1)
                current_state = current_state.squeeze()
                network_value = self.model(current_state).data.item()
                min_value = 0.1 * reward + 0.9 * network_value
                #                print("net value:",network_value)
                if min_value > max_min_value:
                    max_min_value = min_value
                    max_action = action
                    max_state = current_state

            if max_action is None:
                print("follower1 network value is NaN:", max_min_value)
                print("follower1 network error,no max_action")
            self.last_state = max_state
        return max_action

    def rotate(self, state):
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        theta = (state[:, 8] - rot).reshape((batch, -1))
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state
