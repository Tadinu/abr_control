import numpy as np


class Controller:
    """
    The base functions for all controllers

    Parameters
    ----------
    robot_model : class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    """

    def __init__(self, robot_model):
        self.robot_model = robot_model

        self.offset_zeros = np.zeros(3)

    def generate(self, q, dq):
        """
        Generate the torques to apply to robot joints

        Parameters
        ----------
        q : float numpy.array
            joint angles [radians]
        dq : float numpy.array
            the current joint velocities [radians/second]

        """
        raise NotImplementedError
