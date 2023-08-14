from utils.agent import Agent
from utils.state import JointState
from math import atan2
from policy.orca import ORCA


class Human(Agent):
    def __init__(self, config):
        super(Human, self).__init__()
        self.policy = ORCA()
        self.time_step = config.getfloat('env', 'time_step')
        self.policy.time_step = config.getfloat('env', 'time_step')

    def delay_distance(self, action, time_consuming):
        px = self.px + action.vx * time_consuming
        py = self.py + action.vy * time_consuming
        position = [px, py]
        self.px = position[0]
        self.py = position[1]

    def act(self, ob):
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        self.vx = action.vx
        self.vy = action.vy
        self.theta = atan2(self.vy, self.vx)
        return action
