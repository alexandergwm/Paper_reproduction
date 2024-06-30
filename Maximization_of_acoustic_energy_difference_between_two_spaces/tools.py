import numpy as np

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


