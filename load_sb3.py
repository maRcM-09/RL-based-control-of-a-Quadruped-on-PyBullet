# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform

from Helper_Functions import *


# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '102824115106'
log_dir = interm_dir + '123024173552_PD_FLAGRUN_yup'
# log_dir1 = interm_dir + '010325084420_PD_Slopes'
# log_dir2 = interm_dir + '122624051749_CARTESIAN_SLOPES'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
env_config['motor_control_mode'] = "PD"
env_config ['task_env'] = "FLAGRUN"
env_config['observation_space_mode'] = "LR_COURSE_OBS"
env_config["test_flagrun"] =  True 
# env_config['terrain'] = "SLOPES"
# env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
# print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')#, log_dir1, log_dir2

plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0
episode_distance = 0
episode_energy = 0
episode_timestep = 0.001
base_velocity_episode = []
contact_durations = np.zeros(4)

# [TODO] initialize arrays to save data from simulation 
#
x_position = []
y_position = []

time_steps = []
base_positions = []
base_orientations = []
foot_positions = []
foot_contact_booleans = []
base_velocities = []
episode_rewards = []
cost_of_transport = []
# duty_cycles = []
foot_contact_booleans_episode = []


r_trajectories = []
theta_trajectories = []

energy_per_step = 0
motor_torques = []
motor_velocities = []
contact_velocities = np.zeros(4)
got_Initial_position = False

first_time = True
TEST_STEPS = 1200
actions = []

for i in range(TEST_STEPS):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

    actions.append(action)

    if dones:
        # print('episode_reward', episode_reward)
        # print('Final base position', info[0]['base_pos'])
        final_position = base_positions[len(base_positions)-1] # get the final position
        distance = np.linalg.norm(np.array(start_position)-np.array(final_position))
        # duty_cycle = contact_durations / 1000 # 1000 is the episode length
        # duty_cycles.append(duty_cycle)

        mass = np.sum(env.envs[0].env.robot.GetTotalMassFromURDF())
        cost_of_transport.append(episode_energy / (mass * 9.8 * distance))

        avg_velocity = np.mean([vel[0] for vel in base_velocity_episode])
        episode_rewards.append((episode_reward.item(), avg_velocity))
        

        episode_reward = 0
        episode_distance = 0
        episode_energy = 0
        base_velocity_episode = []
        foot_contact_booleans_episode = []
        cost_of_transport = []
        contact_durations = np.zeros(4)
        first_time = True

    # [TODO] save data from current robot states for plots 
    xy_pos = env.envs[0].env.robot.GetBasePosition() 
    x_position.append(xy_pos[0])
    y_position.append(xy_pos[1])

    
    if first_time:
        start_position = env.envs[0].env.robot.GetBasePosition()
        first_time = False

    if env_config['motor_control_mode'] == 'CPG':    
        r_trajectories.append(env.envs[0].env._cpg.get_r())
        theta_trajectories.append(env.envs[0].env._cpg.get_theta())
 

    time_steps.append(i)
    base_positions.append(env.envs[0].env.robot.GetBasePosition())
    base_orientations.append(env.envs[0].env.robot.GetBaseOrientationRollPitchYaw())
    base_velocity = env.envs[0].env.robot.GetBaseLinearVelocity()
    base_velocity_episode.append(base_velocity)
    base_velocities.append(base_velocity)
    foot_positions.append([
        env.envs[0].env.robot.ComputeJacobianAndPosition(leg_id)[1]
        for leg_id in range(4)
    ])
    _, _, _, contact_booleans = env.envs[0].env.robot.GetContactInfo()
    foot_contact_booleans.append(contact_booleans)
    foot_contact_booleans_episode.append(contact_booleans)

    # Energy calculation
    MotorTorques = env.envs[0].env.robot.GetMotorTorques()
    MotorVelocities = env.envs[0].env.robot.GetMotorVelocities()
    energy_step = np.sum(np.abs(np.multiply(MotorTorques, MotorVelocities))) * env.envs[0].env._time_step
    episode_energy += energy_step
    energy_per_step = episode_energy
    motor_torques.append(MotorTorques)
    motor_velocities.append(MotorVelocities)
    
    # Distance covered in x direction
    episode_distance += base_velocity[0] * env.envs[0].env._time_step

    # Update contact durations
    contact_durations = np.sum(np.array(foot_contact_booleans_episode),axis=0)


# print(actions)

end_position =  env.envs[0].env.robot.GetBasePosition()
distance_traveled = np.linalg.norm(np.array(end_position) - np.array(start_position))

# Compute COT
mass = np.sum( env.envs[0].env.robot.GetTotalMassFromURDF())
COT = energy_per_step / (mass * 9.8 * distance_traveled)

# Compute duty cycle and step durations
foot_contact_booleans = np.array(foot_contact_booleans)
duty_cycles, duty_ratios = calculate_duty_cycle(foot_contact_booleans)

step_durations = []
for leg_id in range(4):
    changes = np.diff(foot_contact_booleans[:, leg_id].astype(int))
    step_indices = np.where(changes != 0)[0]
    step_durations.append(np.diff(step_indices) * env.envs[0].env._time_step)

# Convert to numpy arrays
if env_config['motor_control_mode'] == 'CPG':  
    r_trajectories = np.array(r_trajectories)
    theta_trajectories = np.array(theta_trajectories)

##################################################### 
# PLOTS
#####################################################

print(f"Cost of Transport: {COT:.4f}")
print(f"Duty Cycles for legs [0,1,2,3]: {duty_cycles}")
print(f"Duty Ratios for legs [0,1,2,3]: {duty_ratios}")
for i, durations in enumerate(step_durations):
    print(f"Average step Durations for Leg {i}: {np.mean(durations)}")


# Convert lists to numpy arrays
base_positions = np.array(base_positions)
base_orientations = np.array(base_orientations)
base_velocities = np.array(base_velocities)
foot_positions = np.array(foot_positions)
foot_contact_booleans = np.array(foot_contact_booleans)

# duty_cycles = np.array(duty_cycles)
cost_of_transport = np.array(cost_of_transport)

# Print Metrics
# if env_config['task_env'] != 'FLAGRUN':
#     for i, (reward, avg_vel) in enumerate(episode_rewards[:-1]):
print('Episode reward and average velocity:', episode_rewards)


x1 = 100
x2 = 500
plot = False
if plot:
    if env_config['motor_control_mode'] == 'CPG':
        plot_CPG_states(r_trajectories, theta_trajectories, time_steps, x1, x2)
        plot_actionsCPG1(actions, time_steps, x1, x2)
        plot_actionsCPG(actions, time_steps, x1, x2)
    else:
        plot_actions(actions, time_steps, x1=0, x2=30)
        # plot_actions1(actions, time_steps, x1=0, x2=30)
        # plot_actions2(actions, time_steps, x1=0, x2=30)

    plot_Base_velocity(time_steps, base_velocities, x1, x2)
    plot_Contact_Forces(time_steps, foot_contact_booleans, x1, x2)
    plot_Foot_heights (time_steps, foot_positions, x1, x2)  
    plot_Orientation(time_steps, base_orientations, x1, x2)

    if env_config['task_env'] == 'FLAGRUN':
        plot_traj(x_position, y_position)
