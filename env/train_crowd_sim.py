#!/usr/bin/env python
import logging
import gym
import matplotlib.lines as mlines
import numpy as np
# import rvo2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from matplotlib import patches
from numpy.linalg import norm
from utils.human import Human
from utils.info import *
from math import pow, pi, sqrt, atan2,sin,cos,e
from utils.action import ActionXY
from utils.Loging import Log
logging = Log(__name__).getlog()

class TrainCrowdSim(gym.Env):
    metadata = {'render.modes': ['video']}
    def __init__(self):
        self.time_limit = 40
        self.time_step = 0.25
        self.robot = None
        self.follower1 = None
        self.follower2 = None
        self.humans = None
        self.run_env = 'train'
        self.global_time = None
        self.human_times = None

        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = 6
        # for visualization
        self.states = None
        self.human_actions = None
        self.action_flag = True

        self.soft_dist = 0.8
        self.fol_soft_dist = 0.8
        self.pre_dmin = 0
        self.goal_positions = []
        self.config = None

    def configure(self,config):
        self.config = config
        self.time_step = config.getfloat('env','time_step')
    def set_robot(self, robot):
        self.robot = robot

    def set_follower(self, follower1, follower2):
        self.follower1 = follower1
        self.follower2 = follower2

    def set_robot_goal_position(self,gx,gy):
        self.robot.gx = gx
        self.robot.gy = gy
        theta_obj = atan2(self.robot.gy-self.robot.py,self.robot.gx-self.robot.px)
        self.follower1.gx, self.follower1.gy= 1.3*cos(theta_obj+(5*pi)/6) + self.robot.px, 1.3*sin(theta_obj+(5*pi)/6) + self.robot.py
        self.follower2.gx, self.follower2.gy= 1.3*cos(theta_obj-(5*pi)/6) + self.robot.px, 1.3*sin(theta_obj-(5*pi)/6) + self.robot.py

        self.goal_positions.append([gx,gy])

    def reset(self):
        self.human_px = [2.3, 2.2, 3.7, 4.9, 5.2, 8.8]
        self.human_py = [-0.1, 2.8, 0.7,-1.5,-4.3, -3.2]
        self.human_gx = [-1.0, 6.7, 8.2, 7.4, 8.4, 12.3]
        self.human_gy = [1.6, 4.0,-0.4, 2.3, -2.5, -4.5]

        self.global_time = 0
        self.pre_dmin = 0
        self.MSE_Theta = 0
        self.MSE_V = 0
        self.MSE_Flag = 0
        self.flag = 0
        self.theta_list = []
        self.V_list = []
        self.sum_V = 0
        self.pre_theta = 0.7
#        self.pre_thetaF1 = 0.7
#        self.pre_thetaF2 = 0.7
        self.action_list = [ActionXY(0, 0)]
        self.goal_positions = []

        if self.robot is None:
            raise AttributeError('robot has to be set!')

        else:
            self.robot.set(0.0,0.0,0.05,0.05, 0, 0, np.pi / 2)
            self.follower1.set(-1.5,-1.4,-1.1,-0.825, 0, 0, np.pi / 2)
            self.follower2.set(0.6,-1.8,0.33,-1.475, 0, 0, np.pi / 2)
            # print("reset Robot position:", self.robot.px, self.robot.py)
            # print("reset Follower1 position:", self.follower1.px, self.follower1.py)
            # print("reset Follower2 position:", self.follower2.px, self.follower2.py)
            self.human_num = 6
            self.humans = [Human(self.config) for _ in range(self.human_num)]
            self.humans[0].set(self.human_px[0], self.human_py[0], self.human_gx[0], self.human_gy[0], 0, 0, np.pi / 2)
            self.humans[1].set(self.human_px[1], self.human_py[1], self.human_gx[1], self.human_gy[1], 0, 0, np.pi / 2)
            self.humans[2].set(self.human_px[2], self.human_py[2], self.human_gx[2], self.human_gy[2], 0, 0, np.pi / 2)
            self.humans[3].set(self.human_px[3], self.human_py[3], self.human_gx[3], self.human_gy[3], 0, 0, np.pi / 2)
            self.humans[4].set(self.human_px[4], self.human_py[4], self.human_gx[4], self.human_gy[4], 0, 0, np.pi / 2)
            self.humans[5].set(self.human_px[5], self.human_py[5], self.human_gx[5], self.human_gy[5], 0, 0, np.pi / 2)
        self.states = list()
        ob = [human.get_observable_state() for human in self.humans]
        return ob

    def onestep_lookahead(self, action):
        Rvx = action.vx
        Rvy = action.vy
        done = False
        reward = 0
        info = Nothing()
        dmin = float('inf')
        collision = False
        now_theta = round(atan2(Rvy,Rvx),3)

        if self.action_flag == True:
            self.human_actions = []
            for human in self.humans:
                ob0 = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
                self.human_actions.append(human.act(ob0))
            self.action_flag = False

        ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, self.human_actions)]
        # print("pre action:",Rvx,Rvy)
        rx = self.robot.px + Rvx * self.time_step
        ry = self.robot.py + Rvy * self.time_step
        # print("px,py:",self.robot.px,self.robot.py,Rvx,Rvy)
        obj_d = norm([self.robot.px+Rvx*self.time_step-self.robot.gx,self.robot.py+Rvy*self.time_step-self.robot.gy])
        for i, human in enumerate(self.humans):
            px = human.px + human.vx * self.time_step
            py = human.py + human.vy * self.time_step
            closest_dist = norm([rx-px,ry-py])
            if closest_dist < dmin:
                dmin = closest_dist
            if closest_dist < 0.6:
                collision = True

        end_position = np.array(self.robot.compute_position(action, self.time_step))
        robot_goal_dist = norm(end_position - np.array(self.robot.get_goal_position()))
        reaching_goal =  robot_goal_dist < 0.1

        if dmin < self.soft_dist:  #0.6 + 0.6
            reward += 2/(obj_d+0.1) - 2 * obj_d - abs(20/(dmin - 0.5)) + 100*(dmin-self.pre_dmin)
        else:
            reward += 1/(obj_d+0.2) - 2 * obj_d

        if self.global_time >= self.time_limit:
            reward -= 20
            done = True
            info = Timeout()
        elif collision:
            reward -= 20
            done = True
            info = Collision()
        elif reaching_goal:
            reward += 10
            done = False
            info = ReachGoal()
        else:
            reward -= 0.2
            done = False
            info = Nothing()
        if rx >= 12.05 and robot_goal_dist < 0.2:
            reward += 20
            done = True
            info = ReachGoal()
        return ob, reward, done, info

    def onestep_lookahead1(self, action1):
        FFvx = action1.vx
        FFvy = action1.vy
        reward1 = 0
        collision1 = False
        done = False
        info = Nothing()
        dmin1 = float('inf')
        collision = False
        theta_obj = atan2(self.robot.gy-self.robot.py,self.robot.gx-self.robot.px)
        rx = self.robot.px + self.robot.vx * self.time_step
        ry = self.robot.py + self.robot.vy * self.time_step
        best_x1, best_y1= 1.3*cos(theta_obj+(5*pi)/6) + rx, 1.3*sin(theta_obj+(5*pi)/6) + ry
        ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, self.human_actions)]

        r1x = self.follower1.px + FFvx * self.time_step
        r1y = self.follower1.py + FFvy * self.time_step
        r2x = self.follower2.px
        r2y = self.follower2.py
        follower1_robot_dist = norm([r1x-rx,r1y-ry])
        follower1ToFollower2_dist = norm([r2x-r1x,r2y-r1y])
        follower1_best_dist = norm([best_x1-r1x,best_y1-r1y])
        for i, human in enumerate(self.humans):
            px = human.px + human.vx * self.time_step
            py = human.py + human.vy * self.time_step
            closest_dist1 = norm([r1x-px,r1y-py])
            if closest_dist1 < dmin1:
                dmin1 = closest_dist1

        if follower1_robot_dist < 0.6 or dmin1 < 0.6 or follower1ToFollower2_dist < 0.6:
            collision1 = True

        if dmin1 <= self.fol_soft_dist and follower1_robot_dist <= 1.0 and follower1ToFollower2_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - abs(8/(dmin1 - 0.55)) - abs(5/(follower1ToFollower2_dist-0.5)) - abs(6/(follower1_robot_dist-0.5))
        elif dmin1 <= self.fol_soft_dist and follower1_robot_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - abs(8/(dmin1 - 0.55)) - abs(6/(follower1_robot_dist-0.5))
        elif dmin1 <= self.fol_soft_dist and follower1ToFollower2_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - abs(8/(dmin1 - 0.55)) - abs(5/(follower1ToFollower2_dist-0.5))
        elif follower1_robot_dist <= 1.0 and follower1ToFollower2_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - abs(5/(follower1ToFollower2_dist-0.5))- abs(6/(follower1_robot_dist-0.5))
        elif dmin1 <= self.fol_soft_dist :
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - abs(8/(dmin1 - 0.55))
        elif follower1_robot_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - abs(6/(follower1_robot_dist-0.5))
        elif follower1ToFollower2_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - abs(5/(follower1ToFollower2_dist-0.5))
        else:
            reward1 += 2/(follower1_best_dist + 0.1) - 4 * follower1_best_dist * follower1_best_dist

        if follower1_best_dist < 0.1:
            reward1 += 10
            info = ReachGoal()
        if collision1:
            reward1 -= 20
            done = True
            info = Collision()
        return ob, reward1, done, info

    def onestep_lookahead2(self, action2):
        SFvx = action2.vx  #Second Follower
        SFvy = action2.vy
        collision2 = False
        reward2 = 0
        done = False
        reward = 0
        info = Nothing()
        dmin2 = float('inf')
        collision = False
        theta_obj = atan2(self.robot.gy-self.robot.py, self.robot.gx-self.robot.px)
        rx = self.robot.px + self.robot.vx * self.time_step
        ry = self.robot.py + self.robot.vy * self.time_step
        best_x2, best_y2 = 1.3*cos(theta_obj-(5*pi)/6) + rx, 1.3*sin(theta_obj-(5*pi)/6) + ry
        ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, self.human_actions)]

        r1x = self.follower1.px + self.follower1.vx *self.time_step
        r1y = self.follower1.py + self.follower1.vy *self.time_step
        r2x = self.follower2.px + float(SFvx * self.time_step)
        r2y = self.follower2.py + float(SFvy * self.time_step)
        follower2_best_dist = norm([best_x2-r2x,best_y2-r2y])
        follower2_robot_dist = norm([r2x-rx,r2y-ry])
        follower1_follower2_dist = norm([r2x-r1x,r2y-r1y])
        for i, human in enumerate(self.humans):
            px = human.px + human.vx * self.time_step
            py = human.py + human.vy * self.time_step
            closest_dist2 = norm([r2x-px,r2y-py])
            if closest_dist2 < dmin2:
                dmin2 = closest_dist2

        if follower1_follower2_dist < 0.6 or follower2_robot_dist < 0.6 or dmin2 < 0.6:
            collision2 = True

        if dmin2 <= self.fol_soft_dist and follower2_robot_dist <= 1.0 and follower1_follower2_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - abs(8/(dmin2 - 0.55)) - abs(5/(follower1_follower2_dist-0.5)) - abs(6/(follower2_robot_dist-0.5))
        elif dmin2 <= self.fol_soft_dist and follower2_robot_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - abs(8/(dmin2 - 0.55)) - abs(6/(follower2_robot_dist-0.5))
        elif dmin2 <= self.fol_soft_dist and follower1_follower2_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - abs(8/(dmin2 - 0.55)) - abs(5/(follower1_follower2_dist-0.5))
        elif follower2_robot_dist <= 1.0 and follower1_follower2_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - abs(5/(follower1_follower2_dist-0.5)) - abs(6/(follower2_robot_dist-0.5))
        elif dmin2 <= self.fol_soft_dist :
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - abs(8/(dmin2 - 0.55))
        elif follower2_robot_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - abs(6/(follower2_robot_dist-0.5))
        elif follower1_follower2_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - abs(5/(follower1_follower2_dist-0.5))
        else:
            reward2 += 2/(follower2_best_dist + 0.1) - 4 * follower2_best_dist * follower2_best_dist

        if follower2_best_dist < 0.1:
            reward2 += 10
            info = ReachGoal()
        if collision2:
            reward2 -= 20
            done = True
            info = Collision()
        return ob, reward2, done, info


    def step(self, action, update=True):
        # print("Robot position:",self.robot.px,self.robot.py)
        # print("Follower1 position:",self.follower1.px,self.follower1.py)
        # print("Follower2 position:",self.follower2.px,self.follower2.py)
        collision,collision1,collision2 = False,False,False
        reward,reward1,reward2 = 0,0,0
        info = Nothing()

        obj_d = sqrt(pow(self.robot.px+action[0].vx*self.time_step-self.robot.gx,2)+pow(self.robot.py+action[0].vy*self.time_step-self.robot.gy,2))
        theta_obj = atan2(self.robot.gy-self.robot.py,self.robot.gx-self.robot.px)
        dmin, dmin1, dmin2 = float('inf'), float('inf'), float('inf')
        rx = self.robot.px + action[0].vx * self.time_step
        ry = self.robot.py + action[0].vy * self.time_step
        r1x = self.follower1.px + action[1].vx * self.time_step
        r1y = self.follower1.py + action[1].vy * self.time_step
        r2x = self.follower2.px + action[2].vx * self.time_step
        r2y = self.follower2.py + action[2].vy * self.time_step
        best_x1, best_y1 = 1.3*cos(theta_obj+(5*pi)/6) + rx, 1.3*sin(theta_obj+(5*pi)/6) + ry
        best_x2, best_y2 = 1.3*cos(theta_obj-(5*pi)/6) + rx, 1.3*sin(theta_obj-(5*pi)/6) + ry

        follower1_robot_dist = norm([r1x-rx, r1y-ry])
        follower1_best_dist = norm([(best_x1-r1x), (best_y1-r1y)])
        follower1ToFollower2_dist =  norm([r1x-self.follower2.px, r1y-self.follower2.py])

        follower2_best_dist = norm([(best_x2-r2x), (best_y2-r2y)])
      #  print("(",r2x,r2y,")",rx,ry)
        follower2_robot_dist = norm([r2x-rx, r2y-ry])
        follower1_follower2_dist = norm([r2x-r1x, r2y-r1y])

        for i, human in enumerate(self.humans):
            px = human.px + human.vx * self.time_step
            py = human.py + human.vy * self.time_step
            closest_dist = norm([rx-px, ry-py])
            closest_dist1 = norm([r1x-px, r1y-py])
            closest_dist2 = norm([r2x-px, r2y-py])
            if closest_dist < dmin:
                dmin = closest_dist

            if closest_dist1 < dmin1:
                dmin1 = closest_dist1

            if closest_dist2 < dmin2:
                dmin2 = closest_dist2

        end_position = np.array(self.robot.compute_position(action[0], self.time_step))
        robot_goal_dist = norm(end_position - np.array(self.robot.get_goal_position()))
        reaching_goal = robot_goal_dist < 0.1

        if dmin < 0.6:
            collision = True

        if dmin < self.soft_dist:
            reward = 1/(obj_d+0.2) - 2 * obj_d - abs(20/(dmin - 0.5)) + 100*(dmin-self.pre_dmin)
        else:
            reward = 2/(obj_d+0.1) - 2 * obj_d

        if self.global_time >= self.time_limit:
            reward -= 20
            done = True
            info = Timeout()
        elif collision:
            reward -= 20
            done = True
            info = Collision()
            print("Robot collision Human!!!")
        elif reaching_goal:
            reward += 10
            done = False
            info = ReachGoal()
        else:
            reward -= 0.2
            done = False
            info = Nothing()

        if follower1_robot_dist < 0.6:
            print("Follower1 collision Robot !!!")
            collision1 = True
        if follower1ToFollower2_dist < 0.6:
            print("Follower1 collision Follower2 !!!")
            collision1 = True
        if dmin1 < 0.6:
            print("Follower1 collision Human !!!")
            collision1 = True

        if follower1_follower2_dist < 0.6:
            print("Follower2 collision Follower1 !!!")
            collision2 = True
        if follower2_robot_dist < 0.6:
            print("Follower2 collision Robot !!!")
            collision2 = True
        if dmin2 < 0.6:
            print("Follower2 collision Human !!!")
            collision2 = True

        if dmin1 <= self.fol_soft_dist and follower1_robot_dist <= 1.0 and follower1ToFollower2_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - abs(8/(dmin1 - 0.55)) - 5/(follower1ToFollower2_dist-0.5) - 6/(follower1_robot_dist-0.5)
        elif dmin1 <= self.fol_soft_dist and follower1_robot_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - 8/(dmin1 - 0.55) - 6/(follower1_robot_dist-0.5)
        elif dmin1 <= self.fol_soft_dist and follower1ToFollower2_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - 8/(dmin1 - 0.55) - 5/(follower1ToFollower2_dist-0.5)
        elif follower1_robot_dist <= 1.0 and follower1ToFollower2_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - 5/(follower1ToFollower2_dist-0.5)- 6/(follower1_robot_dist-0.5)
        elif dmin1 <= self.fol_soft_dist :
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - 8/(dmin1 - 0.55)
        elif follower1_robot_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - 6/(follower1_robot_dist-0.5)
        elif follower1ToFollower2_dist <= 1.0:
            reward1 += 2/(follower1_best_dist + 0.1) - 2 * follower1_best_dist * follower1_best_dist - 5/(follower1ToFollower2_dist-0.5)
        else:
            reward1 += 2/(follower1_best_dist + 0.1) - 4 * follower1_best_dist * follower1_best_dist

        if follower1_best_dist < 0.1:
            reward1 += 10
        if collision1:
            done = True
            info = Collision()
            reward1 -= 20

        if dmin2 <= self.fol_soft_dist and follower2_robot_dist <= 1.0 and follower1_follower2_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - 8/(dmin2 - 0.55) - 5/(follower1_follower2_dist-0.5) - 6/(follower2_robot_dist-0.5)
        elif dmin2 <= self.fol_soft_dist and follower2_robot_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - 8/(dmin2 - 0.55) - 6/(follower2_robot_dist-0.5)
        elif dmin2 <= self.fol_soft_dist and follower1_follower2_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - 8/(dmin2 - 0.55) - 5/(follower1_follower2_dist- 0.5)
        elif follower2_robot_dist <= 1.0 and follower1_follower2_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - 5/(follower1_follower2_dist-0.5) - 6/(follower2_robot_dist-0.5)
        elif dmin2 <= self.fol_soft_dist :
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - 8/(dmin2 - 0.55)
        elif follower2_robot_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - 6/(follower2_robot_dist-0.5)
        elif follower1_follower2_dist <= 1.0:
            reward2 += 2/(follower2_best_dist + 0.1) - 2 * follower2_best_dist * follower2_best_dist - 5/(follower1_follower2_dist-0.5)
        else:
            reward2 += 2/(follower2_best_dist + 0.1) - 4 * follower2_best_dist * follower2_best_dist

        if follower2_best_dist < 0.1:
            reward2 += 10
        if collision2:
            reward2 -= 20
            done = True
            info = Collision()
        if rx >= 12.05 and robot_goal_dist < 0.2:
            info = ReachGoal()
            reward += 20
            done = True

        if update:
            if self.run_env == 'test':
                self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],self.follower1.get_full_state(),self.follower2.get_full_state()])

            # update all agents
            self.robot.step(action[0])
            self.follower1.step(action[1])
            self.follower2.step(action[2])
            for i, human_action in enumerate(self.human_actions):
                self.humans[i].step(human_action)
            self.action_flag = True
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                if human.reached_destination():
                    self.human_px[i],self.human_gx[i] = self.human_gx[i],self.human_px[i]+np.random.uniform(-0.1,0.1)
                    self.human_py[i],self.human_gy[i] = self.human_gy[i],self.human_py[i]-np.random.uniform(-0.1,0.1)
                    self.humans[i].set(self.human_px[i], self.human_py[i], self.human_gx[i], self.human_gy[i], 0, 0, np.pi / 2)

            # compute the observation
            ob = [human.get_observable_state() for human in self.humans]
            self.pre_dmin = dmin
            v = sqrt(pow(self.robot.vx,2)+pow(self.robot.vy,2))
            self.sum_V += v
            self.V_list.append(v)
            self.MSE_Flag += 1
            current_theta = round(atan2(self.robot.vy,self.robot.vx),3)
#            current_theta1 = round(atan2(self.follower1.vy,self.follower1.vx),3)
#            current_theta2 = round(atan2(self.follower2.vy,self.follower2.vx),3)
            self.theta_list.append(current_theta)
            self.MSE_Theta += pow(current_theta-self.pre_theta,2)
            self.pre_theta = current_theta
#            self.pre_thetaF1 = current_theta1
#            self.pre_thetaF2 = current_theta2
            self.action_list.append(ActionXY(self.robot.vx, self.robot.vy))
        else:
            ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, self.human_actions)]

        return ob, reward, done, info, reward1, reward2

    def render(self, mode='video', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        plt.rc('font',family='Times New Roman', size=16)
        plt.rcParams["font.family"] = "Times New Roman"

        x_offset = 0.25
        y_offset = 0.1
        # cmap = plt.cm.get_cmap('Dark2', 10)
        cmap = 'rosybrown'
        robot_color = 'lime'
        goal_color = 'red'
        arrow_color = 'black'
        path_color = 'blue'
        danger_color = 'oldlace'
        goalposition = 'teal'
        traj_color = 'gainsboro'
        follower_color = 'green'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'video':
            fig, ax = plt.subplots(figsize=(20, 16))
            ax.tick_params(labelsize=16)
            #ax.set_xlim(-2, 14)
            #ax.set_ylim(6, 8)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            x = np.arange(0, 12, 0.01)
            y = 5 * np.sin((x*pi)/6)
            ax.plot(x, y, color='gray')

            follower1_positions = [self.states[i][2].position for i in range(len(self.states))]
            follower2_positions = [self.states[i][3].position for i in range(len(self.states))]
            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            humans_danger_velocity = [[norm(self.states[i][1][j].velocity) for j in range(len(self.humans))] for i in range(len(self.states))]
            humans_danger_radius = [[(0.6+1.8*humans_danger_velocity[i][j]) for j in range(len(self.humans))] for i in range(len(self.states))]
            humans_danger_theta = [[(atan2(self.states[i][1][j].vy,self.states[i][1][j].vx)) for j in range(len(self.humans))] for i in range(len(self.states))]
            humans_danger_thetasize = [[((15*pi/8)*pow(e,-2.5*humans_danger_velocity[i][j])+pi/8) for j in range(len(self.humans))] for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 1 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], 0.03, fill=True, color=path_color)
                    follower1 = plt.Circle(follower1_positions[k], 0.03, fill=True, color=follower_color)
                    follower2 = plt.Circle(follower2_positions[k], 0.03, fill=True, color=follower_color)
                    humans = [plt.Circle(human_positions[k][i], 0.03, fill=True, color=cmap)
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                #if global_time % 10 == 0 or k == len(self.states) - 1:
                if k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=10) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=path_color, ls='solid',zorder = 3)
                    fol1_direction = plt.Line2D((self.states[k - 1][2].px, self.states[k][2].px),
                                               (self.states[k - 1][2].py, self.states[k][2].py),
                                               color=follower_color, ls='solid',zorder = 3)
                    fol2_direction = plt.Line2D((self.states[k - 1][3].px, self.states[k][3].px),
                                               (self.states[k - 1][3].py, self.states[k][3].py),
                                               color=follower_color, ls='solid',zorder = 3)
                    ax.add_artist(nav_direction)
                    ax.add_artist(fol1_direction)
                    ax.add_artist(fol2_direction)

            goal = mlines.Line2D([12.05], [0.1], color=goal_color, marker='*', linestyle='None', markersize=14, label='Goal')
            robotpath = plt.Circle((0,0), 0.1, fill=True, color=path_color)
            trajectory = plt.Circle((0,0), 0.1, fill=True, color=traj_color)
            humanpath = plt.Circle((0,0), 0.1, fill=True, color=cmap)
            danger_area = plt.Circle((0,0), 0.1, fill=True, color=danger_color)
            goal_position = plt.Circle((0,0), 0.1, fill=True, color=goalposition)
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            follower1 = plt.Circle(follower1_positions[0], self.robot.radius, fill=True, color=follower_color)
            follower2 = plt.Circle(follower2_positions[0], self.robot.radius, fill=True, color=follower_color)
            ax.add_artist(follower1)
            ax.add_artist(follower2)
            ax.add_artist(goal)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=True,color='orange') for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i), color='black', fontsize=10) for i in range(len(self.humans))]

            human_dangers = []
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

                v = sqrt(pow(self.states[0][1][i].vx,2)+pow(self.states[0][1][i].vy,2))
                theta0 = atan2(self.states[0][1][i].vy,self.states[0][1][i].vx)
                theta = (15*pi/8)*pow(e,-2.2*v)+pi/8
                r = 0.6 + 1.5*v
                human_danger=Wedge((human.center[0],human.center[1]),r,(theta0-theta/2)*180/pi,(theta0+theta/2)*180/pi,color = danger_color,alpha = 0.5)
                human_dangers.append(human_danger)

            for human_danger in human_dangers:
                ax.add_patch(human_danger)
                plt.axis('equal')

            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(0, 10, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)
            plt.legend([robot,follower1,human,trajectory, robotpath, humanpath,goal_position,goal,danger_area], ['Robot','Follower', 'Human','Trajectory','Robot path','Humans path','Virtual goal','Final goal','Danger area'], fontsize=16, prop={"family":"Times New Roman"})

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                for state in self.states:
                    if i == 0:
                        agent_state = state[0]
                    else:
                        agent_state = state[1][i - 1]
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                    orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                         agent_state.py + radius * np.sin(theta))))
                orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            global ar
            ar = arrows
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def Update(frame_num):
                global ar
                g = {'global_step':0}
                g['global_step'] = frame_num
                robot.center = robot_positions[frame_num]
                follower1.center = follower1_positions[frame_num]
                follower2.center = follower2_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    human_dangers[i].set_center((human.center[0] - 0.4*(humans_danger_radius[frame_num][i]-0.6)*cos(humans_danger_theta[frame_num][i]),
                                                 human.center[1] - 0.4*(humans_danger_radius[frame_num][i]-0.6)*sin(humans_danger_theta[frame_num][i])))
                    human_dangers[i].set_radius(humans_danger_radius[frame_num][i])
                    human_dangers[i].set_theta1((humans_danger_theta[frame_num][i] - humans_danger_thetasize[frame_num][i]/2)*180/pi)
                    human_dangers[i].set_theta2((humans_danger_theta[frame_num][i] + humans_danger_thetasize[frame_num][i]/2)*180/pi)
                    for arrow in ar:
                        arrow.remove()
                    ar = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color, arrowstyle=arrow_style,zorder = 5) for orientation in orientations]
                    for arrow in ar:
                        ax.add_artist(arrow)
#                    if frame_num%9==0:
#                        fig.savefig('Test_Result%d.eps'%frame_num, dpi=600, format='eps')
                if frame_num <len(self.goal_positions):
                    goal = plt.Circle(self.goal_positions[frame_num], 0.04, fill=True, color=goalposition,zorder = 2)
                    ax.add_artist(goal)
                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            anim = animation.FuncAnimation(fig, Update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()

        else:
            raise NotImplementedError
