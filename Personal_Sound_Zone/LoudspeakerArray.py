import numpy as np

class Loudspeaker(object):
    """
    Loudspeaker class
    """
    def __init__(self, name='Generic', version='1.0', direct='Omnidirectional'):
        """
        * Construct
        @ params:
        - name: Nmae of the loudspeaker
        - direct: Directivity of the loudspeaker
        """
        self._spkName = name
        self._ver = version
        self._directivity = direct

    def getname(self):
        """
        * Get the name
        @ return:
        - name of the loudspeaker object
        """
        return self._spkName

    def setname(self, name):
        """
        * Set the name of the loudspeaker
        """
        self._spkName = name

    def getversion(self):
        """
        * Get the version
        """
        print(self._ver)

    def setversion(self, version):
        """
        * Set the version
        """
        self._ver = version


class LoudspeakerArray(Loudspeaker):
    """
    LoudspeakerArray class inherits from loudspeaker class
    """
    def __init__(self, name, type, version, direct):
        """
        * Constructor
        @ params:
        - name: Name of the array
        - type: Type of the array
        - version: Version of the array
        - direct: Directivity of the array
        """
        super(LoudspeakerArray, self).__init__(name, version, direct)
        self.__arraytype = type

    def gettype(self):
        """
        * Get the array type
        """
        return self.__arraytype
    
    def settype(self, type):
        """
        * Set for the array type
        """
        self.__arraytype = type


class EigenmikeEM32(LoudspeakerArray):
    """
    Eigenmike em32 class inherits from the loudspeakerarray class
    """
    def __init__(self):
        super(EigenmikeEM32, self).__init__('Eigenmike 32', 'Rigid Spherical', '17.0', 'Omni')
        self._numelements = 32

        self._thetas = np.array([69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                                 90.0, 125.0, 148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                                 121.0, 159.0, 69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                                 90.0, 125.0, 148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                                 122.0, 159.0]) / 180.0 * np.pi


        self._phis = np.array([0.0, 32.0, 0.0, 328.0, 0.0, 45.0, 69.0, 45.0, 0.0, 315.0,
                               291.0, 315.0, 91.0, 90.0, 90.0, 89.0, 180.0, 212.0, 180.0, 148.0, 180.0,
                               225.0, 249.0, 225.0, 180.0, 135.0, 111.0, 135.0, 269.0, 270.0, 270.0,
                               271.0]) / 180.0 * np.pi
        theta_mask = self._thetas <= (90.0 * np.pi / 180.0)

        filtered_thetas = self._thetas[theta_mask]
        filtered_phis = self._phis[theta_mask]
        self._thetas = filtered_thetas
        self._phis = filtered_phis

        self._radius = 1.8

        self._weights = np.ones(32)

    def returnArrayStruct(self):
        """
        Returns the attributes of the Eigenmike em32 as a struct

        :param self:
        :return: dict object with the name, type, thetas, phis, radius, weights, numelements, directivity
        """
        em32 = {'name': self._micname,
                'thetas': self._thetas,
                'phis': self._phis,
                'radius': self._radius,
                'weights': self._weights,
                'version': self._weights,
                'numelements': self._numelements,
                'directivity': self._directivity}
        return em32

class SphericalArray40(LoudspeakerArray):
    """
    * This class is used to simulate the 3D spherical loudspeaker array which consists of 40 loudspeakers
    """
    def  __init__(self):
        super(SphericalArray40, self).__init__("Spherical 40", "Spherical Array", "1.0", "Omni")
        self._numelements = 40

        # The Cartesian coordinates of the loudspeakers
        spk_positions = np.array([
            [0, 0, 1.8],                   #1
            [0.9463, 0, 1.5312],           #2
            [0.2924, 0.9, 1.5312],         #3
            [-0.7656, 0.5562, 1.5312],     #4
            [-0.7656, -0.5562, 1.5312],    #5
            [0.2924, -0.9, 1.5312],        #6
            [1.61, 0, 0.805],              #7
            [1.2387, 0.9, 0.9463],         #8
            [0.4975, 1.5312, 0.805],       #9
            [-0.4732, 1.4562, 0.9463],     #10
            [-1.3025, 0.9463, 0.805],      #11
            [-1.5312, 0, 0.9463],          #12
            [-1.3025, -0.9463, 0.805],     #13
            [-0.4732, -1.4562, 0.9463],    #14
            [0.4975, -1.5312, 0.805],      #15
            [1.2387, -0.9, 0.9463],        #16
            [1.7119, 0.5562, 0],           #17
            [1.058, 1.4562, 0],            #18
            [0, 1.8, 0],                   #19
            [-1.058, 1.4562, 0],           #20
            [-1.7119, 0.5562, 0],          #21
            [-1.7119, -0.5562, 0],         #22
            [-1.058, -1.4562, 0],          #23
            [0, -1.8, 0],                  #24
            [1.058, -1.4562, 0],           #25
            [1.7119, -0.5562, 0],          #26
            [1.5312, 0, -0.9463],          #27
            [1.3025, 0.9463, -0.805],      #28
            [0.4732, 1.45562, -0.9463],    #29
            [-0.4975, 1.5312, -0.805],     #30
            [-1.2387, 0.9, -0.9463],       #31
            [-1.2387, -0.9, -0.9463],      #32
            [-0.4975, -1.5312, -0.805],    #33
            [0.4732, -1.4562, -0.9463],    #34
            [1.3025, -0.9463, -0.805],     #35
            [0.7656, 0.5562, -1.5312],     #36
            [-0.2924, 0.9, -1.5312],       #37
            [-0.9463, 0, -1.5312],         #38
            [-0.2924, -0.9, -1.5312],      #39
            [0.7656, -0.5562, -1.5312]     #40
        ])

        # Transfer the cartesian coordinates into spherical coordinates
        r = np.linalg.norm(spk_positions, axis=1)
        theta = np.arccos(spk_positions[:,2]/r)
        phi = np.arctan2(spk_positions[:,1], spk_positions[:,0])
        
        self._thetas = theta
        self._phis = phi
        self._radius = 1.8
        self._weights = np.ones(self._numelements)
        self._directivity = "omni"

    
    def returnArraySturct(self):
        """
        * This function retruns the attributes of the 3D spherical array as a struct
        @ return:
        - dict object with the name, type, thetas, phis, radius, weights, numelements, directivity
        """
        SphericalArray40 = {'name': self._spkName,
                            'thetas': self._thetas,
                            'phis': self._phis,
                            'radius': self._radius,
                            'weights': self._weights,
                            'version': self._version,
                            'numelements': self._numelements,
                            'directivity': self._directivity}
        return SphericalArray40 

class LineArray8(LoudspeakerArray):
    """
    * This class is used to simulate the 2D line loudspeaker array which consists of 10 loudspeakers
    """
    def __init__(self):
        super(LineArray8, self).__init__("Line 8", "Line Array", "1.0", "Omni")
        self._numelements = 8
        

        spk_positions = np.array([
            [1.5, 1.65, 2.0],    # [y, x, z]
            [1.5, 1.75, 2.0],
            [1.5, 1.85, 2.0],
            [1.5, 1.95, 2.0],
            [1.5, 2.05, 2.0],
            [1.5, 2.15, 2.0],
            [1.5, 2.25, 2.0],
            [1.5, 2.35, 2.0]
        ])
        self.spk_positions = spk_positions
        self._directivity = "omni"
    
    def returnArraySturct(self):
        """
        * This function retruns the attributes of the 2D line array as a struct
        @ return:
        - dict object with the name, type, thetas, phis, radius, weights, numelements, directivity
        """
        LineArray8 = {'name': self._spkName,
                      'spk_positions':self.spk_positions,
                      'version':self._version,
                      'numelements':self._numelements,
                      'directivity':self._directivity}
        return LineArray8

class CircularArray10(LoudspeakerArray):
    """
    * This class is used to simulate the 2D circular loudspeaker array which consists of 10 loudspeakers
    """
    def __init__(self):
        super(CircularArray10, self).__init__("Circular 10", "Planar Array", "1.0", "Omni")
        self._numelements = 10

        # The cartesian coordinates of the loudspeakers
        spk_positions = np.array([
            [1.7119, 0.5562, 0],           #17
            [1.058, 1.4562, 0],            #18
            [0, 1.8, 0],                   #19
            [-1.058, 1.4562, 0],           #20
            [-1.7119, 0.5562, 0],          #21
            [-1.7119, -0.5562, 0],         #22
            [-1.058, -1.4562, 0],          #23
            [0, -1.8, 0],                  #24
            [1.058, -1.4562, 0],           #25
            [1.7119, -0.5562, 0],          #26
        ])

        self.spk_positions = spk_positions
        r = np.linalg.norm(spk_positions, axis=1)
        theta = np.arccos(spk_positions[:,2]/r)
        phi = np.arctan2(spk_positions[:,1], spk_positions[:,0])
        
        self._thetas = theta
        self._phis = phi
        self._radius = 1.8
        self._weights = np.ones(self._numelements)
        self._directivity = "omni"

    def returnArraySturct(self):
        """
        * This function retruns the attributes of the 2D circular array as a struct
        @ return:
        - dict object with the name, type, thetas, phis, radius, weights, numelements, directivity
        """
        CircularArray10 = {'name': self._spkName,
                            'thetas': self._thetas,
                            'phis': self._phis,
                            'radius': self._radius,
                            'weights': self._weights,
                            'version': self._version,
                            'numelements': self._numelements,
                            'directivity': self._directivity}
        return CircularArray10
    
def print_ArrayInfo(Array):
    name = Array.getname()
    array_type = Array.gettype()
    thetas = Array._thetas
    phis = Array._phis
    radius = Array._radius
    weights = Array._weights
    num_elements = Array._numelements
    directivity = Array._directivity

    # Initialize the list of mic_position in sphercial harmonic domain
    mic_pos_sph = []

    # for each theta and phi, create a new coordinate which contains all parameters to the list of mic_pos
    for theta, phi in zip(thetas, phis):
        mic_pos_sph.append([radius, theta, phi])
    mic_pos_sph = np.array(mic_pos_sph)
    print("Name:", name)
    print("Array Type:", array_type)
    print("Number of elements:", num_elements)
    print("Directivity:", directivity)


if __name__ == "__main__":
    Array = CircularArray10()
    print_ArrayInfo(Array)
   