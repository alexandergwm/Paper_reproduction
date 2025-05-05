import numpy as np
import LoudspeakerArray
import MicrophoneArray

def sph2cart(r, theta, phi):
    """
    * Converts coordinate in spherical coordinates to Cartesian coordinates
    @ params:
    - r: Radius
    - theta: Azimuth angle
    - phi: Inclination angle
    @ returns:
    - Coordinates in Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    car_coor = np.array([x, y, z])
    return car_coor


def calculateDirections(spk_positions, target):
    """
    * Calculate the direction vectors for each loudspeaker to point to the target
    @ params:
    - spk_positions: Array of loudspeaker positions
    - target: The target point [x, y, z]
    """
    directions = target - spk_positions  
    unit_directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]  
    return unit_directions


def generate_circular_spk_array(config):
    """
    * Generate a circular loudspeaker array
    @ params:
    - config: Configuration dictionary
    @ returns:
    - spk_pos_car: Array of loudspeaker positions in Cartesian coordinates
    """
    ## Initialize the loudspeaker array
    spkArray = config.spkArray
    thetas = spkArray._thetas 
    phis = spkArray._phis
    radius = spkArray._radius
    weights = spkArray._weights

    # Initialize the list of spk position in spherical coordinate
    spk_pos_sph = []

    for theta, phi in zip(thetas, phis):
        spk_pos_sph.append([radius, theta, phi])
    spk_pos_sph = np.array(spk_pos_sph)

    spk_pos_car = [sph2cart(r, theta, phi) for r, theta, phi in spk_pos_sph]
    spk_pos_car = np.array(spk_pos_car)
    spk_pos_car = spk_pos_car + 2

    return spk_pos_car

def generate_control_field_circular(config, source_pos_car):
    """
    * Generate a control field
    @ params:
    - config: Configuration dictionary
    - source_pos_car: Array of source positions in Cartesian coordinates
    @ returns:
    - total_field: Array of control field positions in Cartesian coordinates
    """
    ## Initilize the bright field and dark field
    bright_field_x = np.arange(config.bright_field_xrange[0], config.bright_field_xrange[1], config.distance_2mics)
    bright_field_y = np.arange(config.bright_field_yrange[0], config.bright_field_yrange[1], config.distance_2mics)
    bright_field_z = 2.0

    bright_field_x, bright_field_y = np.meshgrid(bright_field_x, bright_field_y)
    bright_field = np.column_stack((bright_field_x.ravel(), bright_field_y.ravel(), np.full(bright_field_x.size, bright_field_z)))
    bright_field[:,1] += 0.4

    dark_field = bright_field.copy()
    dark_field[:, 0] += config.distance_2fields + (config.bright_field_xrange[1] - config.bright_field_xrange[0])
    dark_field[:, 1] -= 0.8
    # If the speaker array is sphercal, planar area will be extended to 3D control area
    if (np.size(np.array([source_pos_car])) / 3 == 40):
        bright_field_list = bright_field.tolist()
        dark_field_list = dark_field.tolist()
        
        for points in bright_field:
            for i in range(-6, 6, 1):
                new_points = [points[0], points[1], points[2] + 0.05 * i]
                bright_field_list.append(new_points)
        
        for points in dark_field:
            for i in range(-6, 6, 1):
                new_points = [points[0], points[1], points[2] + 0.05 * i]
                dark_field_list.append(new_points)
        
        bright_field = np.array(bright_field_list)
        dark_field = np.array(dark_field_list)

    # print(bright_field)
    # print(dark_field)   

                
    bright_field = np.array(bright_field)
    dark_field = np.array(dark_field)
    total_field = np.concatenate((bright_field, dark_field))

    return total_field


def generate_zone_points(angle, num_points, center_distance, control_point_spacing):
    points = []
    center_x = center_distance * np.cos(angle)
    center_y = center_distance * np.sin(angle)
    z = 2

    for i in range(-(num_points//2), num_points//2 + 1):
        dx = i * control_point_spacing * np.cos(angle)
        dy = i * control_point_spacing * np.sin(angle)
        points.append((
            center_x + dx+1.2,
            center_y + dy+2,
            z
        ))
    return points



def generate_control_field_line(config):
    """
    """
    bright_angle = np.deg2rad(-30)  # 亮区角度(弧度)
    dark_angle = np.deg2rad(30)     # 暗区角度(弧度)
    zone_distance = 1.5             # 区中心距离(米)
    control_point_spacing = 0.04    # 控制点间距(米)
    num_control_points = 10         # 每个区的控制点数

    bright_zone = generate_zone_points(bright_angle, num_control_points, zone_distance, control_point_spacing)
    dark_zone = generate_zone_points(dark_angle, num_control_points, zone_distance, control_point_spacing)

    bright_zone = np.array(bright_zone)
    dark_zone = np.array(dark_zone)
    total_field = np.concatenate((bright_zone, dark_zone), axis=0)  # 沿第一个轴（行）连接
    return total_field
    

def generate_line_spk_array(config):
    """
    * Generate a line loudspeaker array
    @ params:
    - config: Configuration dictionary
    @ returns:
    """
    spkArray = config.spkArray
    spk_pos_car = spkArray.spk_positions
    return spk_pos_car

def generate_microphone_array_big(config):
    """
    * Generate a microphone array
    @ params:
    - config: Configuration dictionary
    @ returns:
    - mic_pos_car: Array of microphone positions in Cartesian coordinates
    """
    ## Initialize the microphone array (The whole sampling area bigger than control area)
    micArray = MicrophoneArray.PlanarArray()
    name = micArray.getname()
    array_type = micArray.gettype()
    xs = micArray._x
    ys = micArray._y
    zs = micArray._z
    z = 0
    weights = micArray._weights
    num_elements = micArray._numelements
    directivity = micArray._directivity

    # Initialize the list of mic_position in cartesian coordinates
    mic_pos_car = []    
    # for each theta and phi, create a new coordinate which contains all parameters
    # to the list of mic_ pos
    for x, y in zip(xs, ys):
        mic_pos_car.append([x, y, z])
    mic_pos_car = np.array(mic_pos_car)
    mic_pos_car = mic_pos_car + 2
    # print ("Name:", name)
    # print ("Array Type:", array_type) 
    # print ("Number of Elements:", num_elements)
    # print("Directivity:", directivity)
    
    return mic_pos_car