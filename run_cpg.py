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

""" Run CPG """

import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = True
ADD_JOINT_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

def calculate_duty_cycle(contact_data):
    # num_legs = contact_data.shape[1]
    duty_cycles = []
    duty_ratios = []
    time_step = 0.0001

    for leg_id in range(4):
        # Extract contact data for the current leg
        leg_contact_data = contact_data[:, leg_id]
        """leg contact data basically gathers all the contacts of one foot at a time"""

    # Calculate stance and swing durations
        T_stance = np.sum(leg_contact_data) * time_step
        T_swing = (len(leg_contact_data) - np.sum(leg_contact_data)) * time_step

        # Calculate the duty cycle (T_cycle)
        T_cycle = T_stance + T_swing
        duty_cycles.append(T_cycle)
        duty_ratios.append(T_stance / T_swing)

    return duty_cycles, duty_ratios


# [TODO] initialize data structures to save CPG and robot states
r_trajectories = []
theta_trajectories = []
dr_trajectories = []
dtheta_trajectories = []
state_trajectories = [[] for _ in range(4)]
desired_state_trajectories = [[] for _ in range(4)]
joint_traj = [[] for _ in range(4)]
desired_joint_traj = [[] for _ in range(4)]
base_velocities = []
####### Metrics Calculations ##########

energy_per_step = 0
motor_torques = []
motor_velocities = []
contact_velocities = np.zeros(4)
foot_contact_booleans = []
got_Initial_position = False
############## Sample Gains
# joint PD gains
kp=np.array([60,60,60])
kd=np.array([4,4,4])
# Cartesian PD gains
kpCartesian = np.diag([800]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  if got_Initial_position == False:
     start_position = env.robot.GetBasePosition()
     got_Initial_position = True

  ####### Append CPG States Trajectories ########
  r_trajectories.append(cpg.get_r())
  theta_trajectories.append(cpg.get_theta())
  dr_trajectories.append(cpg.get_dr())
  dtheta_trajectories.append(cpg.get_dtheta())
  ####### Done Adding CPG States Trajectories #####

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities() 

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i,leg_xyz) # [TODO] 
    # Add joint PD contribution to tau for leg i (Equation 4)
    qi = q[3*i:3*(i+1)]
    dqi = dq[3*i:3*(i+1)]
    if ADD_JOINT_PD:
      tau += np.multiply(kd,(-dqi)) + np.multiply(kp,(leg_q-qi)) # [TODO] 


    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO] 
      J, current_xyz = env.robot.ComputeJacobianAndPosition(i)
      # Get current foot velocity in leg frame (Equation 2)
      # [TODO] 
      current_v = J@dqi


      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      
      tau += J.T@(kpCartesian@(leg_xyz-current_xyz) + kdCartesian@(-current_v)) # [TODO]
      
    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

    joint_traj[i].append(qi)
    desired_joint_traj[i].append(leg_q)
    state_trajectories[i].append(env.robot.ComputeJacobianAndPosition(i)[1])
    desired_state_trajectories[i].append(leg_xyz)

    

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 
  torques = env.robot.GetMotorTorques()
  velocities = env.robot.GetMotorVelocities()
  energy_per_step += np.sum(np.abs(torques * velocities)) * TIME_STEP
  motor_torques.append(torques)
  motor_velocities.append(velocities)
  # [TODO] save any CPG or robot states

  # get contact info
  _, _, _, contact_booleans = env.robot.GetContactInfo()
  foot_contact_booleans.append(contact_booleans)

  # Collect base velocity
  base_velocity = env.robot.GetBaseLinearVelocity()
  base_velocities.append(np.array(base_velocity))

end_position = env.robot.GetBasePosition()
distance_traveled = np.linalg.norm(np.array(end_position) - np.array(start_position))

# Compute COT
mass = np.sum(env.robot.GetTotalMassFromURDF())
COT = energy_per_step / (mass * 9.8 * distance_traveled)

# Compute duty cycle and step durations
foot_contact_booleans = np.array(foot_contact_booleans)
duty_cycles, duty_ratios = calculate_duty_cycle(foot_contact_booleans)
step_durations = []
for leg_id in range(4):
    changes = np.diff(foot_contact_booleans[:, leg_id].astype(int))
    step_indices = np.where(changes != 0)[0]
    step_durations.append(np.diff(step_indices) * TIME_STEP)

# Convert to numpy arrays
r_trajectories = np.array(r_trajectories)
theta_trajectories = np.array(theta_trajectories)
dr_trajectories = np.array(dr_trajectories)
dtheta_trajectories = np.array(dtheta_trajectories)
state_trajectories = [np.array(leg) for leg in state_trajectories]
desired_state_trajectories = [np.array(leg) for leg in desired_state_trajectories]
joint_traj = [np.array(joint) for joint in joint_traj]
desired_joint_traj = [np.array(joint) for joint in desired_joint_traj]
base_velocities = np.array(base_velocities)
#################################################### 
# PLOTS
####################################################
# Plot CPG States
plt.figure(figsize=(5, 4))
labels = ['FR', 'FL', 'RR', 'RL']
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(t[0:2000], dr_trajectories[0:2000, i], label=f'Leg {labels[i]}')
    plt.xlabel('Time (s)')
    plt.ylabel('dr')
    plt.legend()
    plt.grid()
plt.suptitle('CPG dr States')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(t[0:2000], dtheta_trajectories[0:2000, i], label=f'Leg {labels[i]}')
    plt.xlabel('Time (s)')
    plt.ylabel('d_Theta')
    plt.legend()
    plt.grid()
plt.suptitle('CPG d_Theta States')
plt.tight_layout()
plt.show()

# Plot Desired vs Actual Foot Positions
plt.figure(figsize=(5, 4))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(t[0:2000], state_trajectories[0][0:2000, i], label=f'Actual Position {labels[i]}')
    plt.plot(t[0:2000], desired_state_trajectories[0][0:2000, i], label=f'Desired Position {labels[i]}', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Foot Position (m)')
    plt.legend()
    plt.grid()
plt.suptitle('X-Y-Z states FR leg')
plt.tight_layout()
plt.show()

# Plot Desired vs Actual Joint Angles
plt.figure(figsize=(5, 4))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(t[0:2000], joint_traj[0][0:2000, i], label=f'Actual Joint {i} FR')
    plt.plot(t[0:2000], desired_joint_traj[0][0:2000, i], label=f'Desired Joint {i} FR', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.legend()
    plt.grid()
plt.suptitle('Joint Angles for FR Leg')
plt.tight_layout()
plt.show()

print(f"Average body velocity in the x-direction: {np.mean(base_velocities[:,0]):.3f}")
print(f"Cost of Transport: {COT:.4f}")
print(f"Duty Cycles for leges [0,1,2,3]: {duty_ratios}")
for i, durations in enumerate(step_durations):
    print(f"Average step Durations for Leg {i}: {np.mean(durations):.3f}")
