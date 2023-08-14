import itertools
import torch
import numpy as np

from policy.policy import Policy
from utils.action import ActionXY
import torch.nn as nn
from utils.state import ObservableState, FullState


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


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp_dims, lstm_hidden_dim):
        super(ValueNetwork, self).__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)

        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class LstmRL(Policy):
    def __init__(self):
        super(LstmRL, self).__init__()
        self.action_space = None
        self.epsilon = 0.3
        self.phase = 'train'
        self.last_state = None
        self.model = None
        self.env = None
        self.time_step = 0.25
        self.model = ValueNetwork(13, 6, [150, 100, 100, 1], 60)
        print('lstm_rl:', self.model)

    def configure(self, config):
        self.time_step = config.getfloat('env', 'time_step')

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_env(self, env):
        self.env = env

    def build_action_space(self):
        speeds = [0.2, 0.5, 0.7, 1.0, 1.2]
        rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        action_space = [ActionXY(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            action_space.append(ActionXY(np.float(speed * np.cos(rotation)), np.float(speed * np.sin(rotation))))
        self.action_space = action_space
        print("leader action_space:", len(self.action_space))

    def transform(self, state):
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]) for human_state in state.human_states],
                                 dim=0)
        state_tensor = self.rotate(state_tensor)
        return state_tensor

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius, state.gx, state.gy,
                                   state.v_pref, state.theta)
        else:
            raise ValueError('Type error')
        return next_state

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

    def predict(self, state):
        def dist(human):
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)

        if self.action_space is None:
            self.build_action_space()

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state])
                                               for next_human_state in next_human_states], dim=0)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                next_state_value = self.model(rotated_batch_input).data.item()
                value = 0.1 * reward + 0.9 * next_state_value
                if value > max_value:
                    max_value = value
                    max_action = action

            if max_action is None:
                print("Robot network value is NaN:", max_value)
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action
