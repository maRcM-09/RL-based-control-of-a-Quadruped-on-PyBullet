import numpy as np
import matplotlib.pyplot as plt

def calculate_duty_cycle(contact_data):
    # num_legs = contact_data.shape[1]
    duty_cycles = []
    duty_ratios = []
    time_step = 0.001

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



def plot_CPG_states(r_trajectories, theta_trajectories, time_steps, x1, x2):
    plt.figure(figsize=(10, 8))
    labels = ['FR', 'FL', 'RR', 'RL']
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(time_steps, r_trajectories[:, i], label=f'Leg {labels[i]}')
        plt.xlabel('Time (s)')
        plt.ylabel('r')
        plt.xlim([x1, x2])
        plt.legend()
        plt.grid()
    plt.suptitle('CPG r States')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(time_steps, theta_trajectories[:, i], label=f'Leg {labels[i]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Theta')
        plt.xlim([x1, x2])
        plt.legend()
        plt.grid()
    plt.suptitle('CPG Theta States')
    plt.tight_layout()
    plt.show()





def plot_Base_velocity(time_steps, base_velocities, x1, x2):
    plt.figure()
    components = ['X', 'Y', 'Z']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(time_steps, base_velocities[:, i], label=f'{components[i]}-velocity')
        plt.xlabel('Timestep')
        plt.ylabel(f'{components[i]} Velocity (m/s)')
        plt.xlim([x1, x2])
        plt.grid()
    plt.suptitle('Base Velocities Over Time')
    plt.tight_layout()
    plt.show()



def plot_Contact_Forces(time_steps, foot_contact_booleans, x1, x2):
    plt.figure()
    j = 1
    for i in range(0,4,2):
        plt.subplot(2, 1, j)
        plt.plot(time_steps[0:300], foot_contact_booleans[0:300, i], label=f'Leg {i}', drawstyle='steps-post')
        plt.plot(time_steps[0:300], foot_contact_booleans[0:300, i+1], label=f'Leg {i}', drawstyle='steps-post')
        plt.xlabel('Timestep')
        plt.ylabel(f'Contact (bool)')
        plt.grid()
        j+=1
    plt.suptitle('Foot Contact Booleans')
    plt.tight_layout()
    plt.show()




def split(x_pos, y_pos, threshold):
    for i in range(len(x_pos) - 1):
        if abs(x_pos[i] - x_pos[i + 1]) > threshold:
            # Split the trajectory at the gap
            split_x = x_pos[:i + 1]
            next_x = x_pos[i + 1:]
            split_y = y_pos[:i + 1]
            next_y = y_pos[i + 1:]
            return split_x, split_y, next_x, next_y

    # If no gap is found, return the full trajectory as one segment
    return x_pos, y_pos, [], []


goal_x = np.arange(np.pi/4, 11, np.pi/2)
goal_y = 0.2 * goal_x * np.sin(2*goal_x)

goal_location = np.array([goal_x[0], goal_y[0]])
goal_location2 = np.array([goal_x[1], goal_y[1]])
goal_location3 = np.array([goal_x[2], goal_y[2]])
goal_location4 = np.array([goal_x[3], goal_y[3]])
goal_location5 = np.array([goal_x[4], goal_y[4]])
goal_location6 = np.array([goal_x[5], goal_y[5]])
goal_location7 = np.array([goal_x[6], goal_y[6]])

def plot_traj(x_pos, y_pos):
    x_segments = []
    y_segments = []
    threshold = 0.5  # Define the threshold for splitting trajectories

    # Keep splitting until no gaps are found
    while len(x_pos) > 0:
        x_split, y_split, x_pos, y_pos = split(x_pos, y_pos, threshold)

        # Append the current segment if it’s non-empty
        if len(x_split) > 0:
            x_segments.append(x_split)
            y_segments.append(y_split)

        # Exit if no more points remain
        if len(x_pos) == 0:
            break

    # Plot each segment
    for i in range(len(x_segments)):
        plt.plot(x_segments[i], y_segments[i], label=f'Episode {i + 1}')

    plt.scatter(0, 0, color='green', marker='o', s=100, label='Initial Position')
    plt.scatter(goal_location[0], goal_location[1], color='red', marker='X', s=100)
    plt.scatter(goal_location2[0], goal_location2[1], color='red', marker='X', s=100)
    plt.scatter(goal_location3[0], goal_location3[1], color='red', marker='X', s=100)
    plt.scatter(goal_location4[0], goal_location4[1], color='red', marker='X', s=100)
    plt.scatter(goal_location5[0], goal_location5[1], color='red', marker='X', s=100)
    plt.scatter(goal_location6[0], goal_location6[1], color='red', marker='X', s=100)
    plt.scatter(goal_location7[0], goal_location7[1], color='red', marker='X', s=100)


    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Trajectory Plot')
    plt.legend()
    plt.show()



def plot_Foot_heights(time_steps, foot_positions, x1, x2):
    plt.figure()
    for leg_id in range(4):
        plt.subplot(4, 1, leg_id + 1)
        plt.plot(time_steps, foot_positions[:, leg_id, 2], label=f'Leg {leg_id}')
        plt.xlabel('Timestep')
        plt.ylabel('Foot Height (m)')
        plt.xlim([x1, x2])
        plt.grid()
    plt.suptitle('Foot Heights Over Time')
    plt.tight_layout()
    plt.show()

def plot_actions(actions, time_steps, x1, x2):
    actions = np.array(actions)  # Ensure actions is a numpy array
    actions = np.squeeze(actions)  # Remove unnecessary dimensions
    num_actions = actions.shape[1]
    plt.figure(figsize=(10, num_actions * 2))
    for i in range(num_actions):
        plt.subplot(num_actions, 1, i + 1)
        plt.plot(time_steps, actions[:, i], label=f'Action {i}')
        plt.xlabel('Timestep')
        plt.ylabel(f'Action {i}')
        plt.xlim([x1, x2])
        plt.grid()
        plt.legend()

    plt.suptitle('Actions Over Time')
    plt.tight_layout()
    plt.show()

def plot_actions1(actions, time_steps, x1, x2):
    actions = np.array(actions)  # Ensure actions is a numpy array
    actions = np.squeeze(actions)  # Remove unnecessary dimensions
    num_actions = actions.shape[1]
    plt.figure(figsize=(10, num_actions * 2))
    for i in range(num_actions//3, 2*num_actions//3, 1):
        plt.subplot(num_actions, 1, i + 1)
        plt.plot(time_steps, actions[:, i], label=f'Action {i}')
        plt.xlabel('Timestep')
        plt.ylabel(f'Action {i}')
        plt.xlim([x1, x2])
        plt.grid()
        plt.legend()

    plt.suptitle('Actions Over Time')
    plt.tight_layout()
    plt.show()


def plot_actions2(actions, time_steps, x1, x2):
    actions = np.array(actions)  # Ensure actions is a numpy array
    actions = np.squeeze(actions)  # Remove unnecessary dimensions
    num_actions = actions.shape[1]
    plt.figure(figsize=(10, num_actions * 2))
    for i in range(2*num_actions//3, num_actions, 1):
        plt.subplot(num_actions, 1, i + 1)
        plt.plot(time_steps, actions[:, i], label=f'Action {i}')
        plt.xlabel('Timestep')
        plt.ylabel(f'Action {i}')
        plt.xlim([x1, x2])
        plt.grid()
        plt.legend()

    plt.suptitle('Actions Over Time')
    plt.tight_layout()
    plt.show()

def plot_actionsCPG1(actions, time_steps, x1, x2):
    actions = np.array(actions)  # Ensure actions is a numpy array
    actions = np.squeeze(actions)  # Remove unnecessary dimensions
    num_actions = actions.shape[1]
    plt.figure(figsize=(10, num_actions * 2))
    for i in range(num_actions//2):
        plt.subplot(num_actions, 1, i + 1)
        plt.plot(time_steps, actions[:, i], label=f'Action {i}')
        plt.xlabel('Timestep')
        plt.ylabel(f'Action {i}')
        plt.xlim([x1, x2])
        plt.grid()
        plt.legend()

    plt.suptitle('Actions Over Time')
    plt.tight_layout()
    plt.show()

def plot_actionsCPG(actions, time_steps, x1, x2):
    actions = np.array(actions)  # Ensure actions is a numpy array
    actions = np.squeeze(actions)  # Remove unnecessary dimensions
    num_actions = actions.shape[1]
    plt.figure(figsize=(10, num_actions * 2))
    for i in range(num_actions//2, num_actions, 1):
        plt.subplot(num_actions, 1, i + 1)
        plt.plot(time_steps, actions[:, i], label=f'Action {i}')
        plt.xlabel('Timestep')
        plt.ylabel(f'Action {i}')
        plt.xlim([x1, x2])
        plt.grid()
        plt.legend()

    plt.suptitle('Actions Over Time')
    plt.tight_layout()
    plt.show()


def plot_Orientation(time_steps, base_orientations, x1, x2):
    plt.figure()
    orientations = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(time_steps, base_orientations[:, i], label=f'{orientations[i]}')
        plt.xlabel('Timestep')
        plt.ylabel(f'{orientations[i]} (rad)')
        plt.xlim([x1, x2])
        plt.grid()
    plt.suptitle('Base Orientation Over Time')
    plt.tight_layout()
    plt.show()

