import numpy as np

from .controller import Controller


class Damping(Controller):
    """Base class for common null space controllers.

    Parameters
    ----------
    robot_model : class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    """

    def __init__(self, robot_model, kv):
        super().__init__(robot_model)

        self.kv = kv

    def generate(self, q, dq):
        """Generates the control signal

        q : np.array
          the current joint angles [radians]
        dq : np.array
          the current joint angle velocity [radians/second]
        """

        # calculate joint space inertia matrix
        M = self.robot_model.M(q=q)
        return np.dot(M, -self.kv * dq)
