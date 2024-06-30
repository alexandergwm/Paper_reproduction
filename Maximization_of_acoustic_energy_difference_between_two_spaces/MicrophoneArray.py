import numpy as np


class Microphone(object):
    """
    Microphone class
    """

    def __init__(self, name='Generic', version='1.0', direct='Omnidirectional'):
        """
        * Constructor

        @ params:
        -name: Name of the microphone
        -direct: Directivity of the microphone (str)
        """
        self._micname = name
        self._ver = version
        self._directivity = direct

    def getname(self):
        """
        * Get the name
        @ returns:
        - Name (str) of the microphone object
        """
        return self._micname

    def setname(self, name):
        """
        * Set for the name attribute
        @ params:
        - name:  Name of the microphone (str)
        """
        self._micname = name

    def getversion(self):
        """
        * Get the version
        @ returns:
        - return: Version(str) of the microphone object
        """
        print(self._ver)

    def setversion(self, version):
        """
        * Set for the version

        @ params:
        - version: Version of the microphone (str)
        """
        self._ver = version

class MicrophoneArray(Microphone):
    """
    * MicrophoneArray class inherits from Microphone class
    """

    def __init__(self, name, typ, version, direct):
        """
        * Constructor

        @ params:
        - name: Name of the array
        - typ: Type of the array
        - version: Version of the array
        - direct: Directivity of components
        """
        super(MicrophoneArray, self).__init__(name, version, direct)
        self._arraytype = None
        self.__arraytype = typ

    def gettype(self):
        """
        * Get for the array type
        """
        return self.__arraytype

    def settype(self, typ):
        """
        * set for array type
        """
        self.__arraytype = typ

class PlanarArray(MicrophoneArray):

    def __init__(self):
        super(PlanarArray, self).__init__('Cube Mic Array', 'Open Spherical','1.0', 'Omni')
        
        mic1 = [-1, -0.74, 0]
        mic_positions = []
        planar_mic_pos = []
        # Generate coordinates from mic2 to mic30
        for i in range(30):
            new_mic = [mic1[0], mic1[1]+0.05*i, mic1[2]]
            mic_positions.append(new_mic)
        mic_positions = np.array(mic_positions)

        for j in range(40):
            for mic in mic_positions:
                new_mic = [mic[0] + 0.05*j, mic[1], mic[2]]
                planar_mic_pos.append(new_mic)
        mic_positions = np.array(planar_mic_pos)

        self._x = mic_positions[:,0]
        self._y = mic_positions[:,1]
        self._z = mic_positions[:,2]
        self._numelements = len(self._x)
        self._weights = np.ones(self._numelements)

    def returnArrayStruct(self):
        """
        * Returns the attributes of the planar mic array as a struct
        @ returns:
        - dict object with the name, type, x, y, z, weights, numelements, directivity
        """
        MicArray = {'name': self._micname,
                'x': self._x,
                'y': self._y,
                'z': self._z,
                'weights': self._weights,
                'version': self._version,
                'numelements': self._numelements,
                'directivity': self._directivity}
        return MicArray



if __name__ == "__main__":
    micArray = PlanarArray()
    name = micArray.getname()
    array_type = micArray.gettype()
    xs = micArray._x
    ys = micArray._y
    zs = micArray._z
    z = 0
    weights = micArray._weights
    num_elements = micArray._numelements
    directivity = micArray._directivity

    mic_pos_car = []
    for x, y in zip(xs, ys):
        mic_pos_car.append([x, y, z])
    mic_pos_car = np.array(mic_pos_car)
    print("Name:", name)
    print("Array Type:", array_type)
    print("x coordinates:", xs)
    print("y coordinates:", ys)
    print("z coordinates:", zs)
    print("Number of elements:", num_elements)

