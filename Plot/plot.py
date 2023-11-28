import numpy as np
import matplotlib.pyplot as plt

def plot(abs_pos, thymio_coords, goal_position, map_global, closest_angle):
    # Plot
    # Plot Map
    plt.imshow(map_global, cmap='binary', origin='upper')
    plt.title("Global Map")
    plt.xlim(0, len(map_global[:,1]))
    plt.ylim(0, len(map_global[1,:]))
    plt.xlabel("X [cm]")
    plt.ylabel("Y [cm]")
    # Plot goal
    plt.scatter(goal_position[0][0], goal_position[0][1], color='red')
    # Plot thymio
    # Rotation matrix
    theta = abs_pos[0][2]
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
    # Thymio
    thymio_coords_rotated = np.dot(thymio_coords, rotation_matrix)
    thymio_X = abs_pos[0][0]+thymio_coords_rotated[:,0]
    thymio_y = abs_pos[0][1]+thymio_coords_rotated[:,1]
    plt.plot(thymio_X, thymio_y)
    # Plot Direction
    if state==1:
        x, y, angle = abs_pos[0][:]
        dx = math.cos(closest_angle)#angle_radians
        dy = math.sin(closest_angle)#closest_angle
        plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=0.1, color='r', width=0.01)


    plt.show()


def configure_ax(ax, max_x, max_y):
    """
    Helper function to create a figure of the desired dimensions & grid
    
    :param max_val: dimension of the map along the x and y dimensions
    :return: the fig and ax objects.
    """
    MAJOR = 10
    MINOR = 1
    
    major_ticks_x = np.arange(0, max_x+1, MAJOR)
    minor_ticks_x = np.arange(0, max_x+1, MINOR)
    major_ticks_y = np.arange(0, max_y+1, MAJOR)
    minor_ticks_y = np.arange(0, max_y+1, MINOR)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,max_y])
    ax.set_xlim([-1,max_x])
    ax.grid(True)

    return ax