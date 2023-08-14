from .train_crowd_sim import TrainCrowdSim
from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='env:CrowdSim',
)

