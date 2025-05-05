import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import torchaudio
import scipy.signal as signal
import librosa.display

def plot_array(spk_pos_car, arrayname):
    """
    * Plot the microphone array in the room
    @ params:
    - spk_pos_car: The cartesian coordinate of the loudspeaker array
    - arrayname
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(spk_pos_car[:,0], spk_pos_car[:,1], spk_pos_car[:,2], c='g', marker='o', label=arrayname)
    # Set the axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    # Show the legend
    ax.legend()
    plt.show()


def verify_shapeofArray(spk_pos_car):
    """
    * Projecting the shape of the array into three planars
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # xy plane
    axs[0].scatter(spk_pos_car[:, 0], spk_pos_car[:, 1], c='g')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Projection onto XY plane')
    axs[0].set_aspect('equal', 'box')

    # xz plane
    axs[1].scatter(spk_pos_car[:,0], spk_pos_car[:,2], c='g')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('Projection onto XZ plane')
    axs[1].set_aspect('equal', 'box')

    # yz plane
    axs[2].scatter(spk_pos_car[:,1], spk_pos_car[:,2], c='g')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_title('Projection onto YZ plane')
    axs[2].set_aspect('equal', 'box')
    
    plt.show()

def plot_total(room_dim, mic_pos_car_list, spk_pos_car_list, control = False):
    """
    * plot the microphone array and loudspeaker array in the room
    @ params:
    - room_dim: the dimensions of the room
    - mic_pos_car_list: The list of the cartesian positions of microphone array
    - spk_pos_car_list: The list of the cartesian positions of loudspeaker array
    - control: decide to plot the control filed or not
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the microphone array
    for i, mic_pos_car in enumerate(mic_pos_car_list):
        ax.scatter(mic_pos_car[:,0], mic_pos_car[:,1], mic_pos_car[:,2], c='g', marker='o', label=f'Microphone Array {i+1}')

    # Plot the loudspeaker array
    for i, spk_pos_car in enumerate(spk_pos_car_list):
        ax.scatter(spk_pos_car[:,0], spk_pos_car[:,1], spk_pos_car[:,2], c='r', marker='x', label=f'Loudspkear Array {i+1}', s=100)

    # Plot the control planar area
    control_field1 = []
    control_temp1 = []
    control_temp2 = []
    control_field2 = []
    control_field_tot = []
    temp = 2
    for i, mic_pos_car_total in enumerate(mic_pos_car_list):
        for mic_pos_car in (mic_pos_car_total):
            if ((mic_pos_car[0] >= -0.8+temp) & (mic_pos_car[0] <= -0.2+temp) & (mic_pos_car[1] >= -0.3+temp) & (mic_pos_car[1] <= 0.3+temp)).all():
                control_temp1.append(mic_pos_car)
            if ((mic_pos_car[0] >= 0.2+temp) & (mic_pos_car[0] <= 0.8+temp) & (mic_pos_car[1] >= -0.3+temp) & (mic_pos_car[1] <= 0.3+temp)).all():
                control_temp2.append(mic_pos_car)

    # If the speaker array is spherical, planar area will be extended to 3D control area
    if (np.size(np.array(spk_pos_car_list))/3 == 40):
        for j, points in enumerate(control_temp1):
            for i in range(-6,6,1):
                new_points = [points[0], points[1], points[2]+0.05*i]
                control_field1.append(new_points)
        for j, points in enumerate(control_temp2):
            for i in range(-6,6,1):
                new_points = [points[0], points[1], points[2]+0.05*i]
                control_field2.append(new_points)
        control_field_tot = [control_field1, control_field2]
    else:
        control_field_tot = [control_temp1, control_temp2]

    if control == True:
        for i, control_points in enumerate(control_field_tot):
            control_points = np.array(control_points)
            ax.scatter(control_points[:,0], control_points[:,1], control_points[:,2], c='blue', marker=',', label=f'Acoustic control space {i+1}')

    # Set axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    # Show the legend
    ax.legend()
    
    # Show the plot
    ax.set_xlim(0, room_dim[0])
    ax.set_ylim(0, room_dim[1])
    ax.set_zlim(0, room_dim[2])
    plt.show()


def plot_loudspeaker_dir(spk_pos_car, directions):
    """
    * Plot loudspeaker directivities
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (pos, dir) in enumerate(zip(spk_pos_car, directions)):
        ax.quiver(pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], length=0.1, normalize=True, color=f"C{i}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Loudspeaker Directivity')
    ax.legend()
    plt.show()
