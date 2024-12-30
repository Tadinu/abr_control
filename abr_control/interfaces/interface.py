from typing_extensions import List
from abr_control.arms import DeviceModel

class Interface:
    """Base class for interfaces

    The purpose of interfaces is to abstract away the API
    and overhead of connection / disconnection etc for each
    of the different systems that can be controlled.
    """

    def __init__(self):
        self.sim = None
        self.dt: float = 0
        self.time: float = 0  # accumulated time over steps
        self.ee_names: List[str] = []
        self.joint_names: List[str] = []
        self.device_models: List[DeviceModel] = []
        pass

    def connect(self, scene_xml_path: str, dt, joint_names=None, camera_id=-1, **kwargs):
        """All initial setup."""

        raise NotImplementedError

    def disconnect(self):
        """Any socket closing etc that must be done to properly shut down"""

        raise NotImplementedError

    def send_forces(self, device_model: DeviceModel, u, use_joint_dyn_addrs=True):
        """Applies the set of torques u to the arm. If interfacing to
        a simulation, also moves dynamics forward one time step.

        u : np.array
            An array of joint torques [Nm]
        """

        raise NotImplementedError

    def send_target_angles(self, device_model: DeviceModel, q):
        """Moves the arm to the specified joint angles

        q : numpy.array
            the target joint angles [radians]
        """

        raise NotImplementedError

    def get_feedback(self, device_model: DeviceModel):
        """Returns a dictionary of the relevant feedback

        Returns a dictionary of relevant feedback to the
        controller. At very least this contains q, dq.
        """

        raise NotImplementedError
