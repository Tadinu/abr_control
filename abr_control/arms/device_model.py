import os.path

import numpy as np
from threading import Lock
from typing_extensions import Dict, Any, Optional
from enum import Enum
import copy
from abr_control.arms.mujoco_model import MujocoModel
class DeviceState(Enum):
    Q = 'Q'
    Q_ACTUATED = 'Q_ACTUATED'
    DQ = 'DQ'
    DQ_ACTUATED = 'DQ_ACTUATED'
    DDQ = 'DDQ'
    EE_XYZ = 'EE_XYZ'
    EE_XYZ_VEL = 'EE_XYZ_VEL'
    EE_QUAT = 'EE_QUAT'
    FORCE = 'FORCE'
    TORQUE = 'TORQUE'
    J = 'JACOBIAN'

class DeviceModel(MujocoModel):
    """
    The Device class encapsulates the device parameters specified in the yaml file
    that is passed to AbrMujoco. It collects data from the simulator, obtaining the 
    desired device states.
    """
    def __init__(self, sim_interface, device_yml: Optional[Dict] = None, xml_file=None, use_sim_state=True):
        super().__init__(sim_interface, xml_file=xml_file, use_sim_state=use_sim_state)
        self.use_sim_state = use_sim_state
        # Assign all of the yaml parameters
        self.name = device_yml['name'] if device_yml else os.path.basename(xml_file)
        self.max_vel = device_yml['max_vel'] if device_yml else None
        self.EE = device_yml['EE'] if device_yml else None
        self.ctrlr_dof_xyz = device_yml['ctrlr_dof_xyz'] if device_yml else None
        self.ctrlr_dof_abg = device_yml['ctrlr_dof_abg'] if device_yml else None #alpha, beta, gamma
        self.ctrlr_dof = np.hstack([self.ctrlr_dof_xyz, self.ctrlr_dof_abg])
        self.start_angles = np.array(device_yml['start_angles']) if device_yml else None
        self.num_gripper_joints = device_yml['num_gripper_joints'] if device_yml else None
        
        # Check if the user specifies a start body for the while loop to terminte at
        self.start_body_id = self.sim_model.name2id(device_yml['start_body'], 'body') if device_yml and self.sim_model else 0
        
        if self.sim_model:
            parent_body_id = lambda child_body_id : self.sim_model.body_parentid[child_body_id]
            # Reference: ABR Control
            # Get the joint ids, using the specified EE / start body 
            # start with the end-effector (EE) and work back to the world body
            body_id = self.sim_model.name2id(self.EE, 'body')
            print(self.EE, ':', body_id, 'parent:', parent_body_id(body_id), self.sim_model.id2name(parent_body_id(body_id), 'body'))
            joint_ids = []
            joint_names = []
            while parent_body_id(body_id) != 0 and parent_body_id(body_id) != self.start_body_id:
                jntadrs_start = self.sim_model.body_jntadr[body_id]
                tmp_ids = []
                tmp_names = []
                for i in range(self.sim_model.body_jntnum[body_id]):
                    tmp_ids.append(jntadrs_start + i)
                    tmp_names.append(self.sim_model.id2name(tmp_ids[-1], 'joint'))
                joint_ids += tmp_ids[::-1]
                joint_names += tmp_names[::-1]
                body_id = parent_body_id(body_id)

            # flip the list so it starts with the base of the arm / first joint
            self.joint_names = joint_names[::-1] if len(joint_names) else []
            self.joint_ids = np.array(joint_ids[::-1]) if len(joint_ids) else []
            if len(self.joint_ids):
                print('joint names: ', self.joint_names)
                print('joint ids: ', self.joint_ids)

                gripper_start_idx = self.joint_ids[-1] + 1
                self.gripper_ids = np.arange(gripper_start_idx, 
                                             gripper_start_idx + self.num_gripper_joints)
                self.joint_ids_all = np.hstack([self.joint_ids, self.gripper_ids])

                # Find the actuator and control indices
                actuator_trnids = self.sim_model.actuator_trnid[:,0]
                self.ctrl_idxs = np.intersect1d(actuator_trnids, self.joint_ids_all, return_indices=True)[1]
                self.actuator_trnids = actuator_trnids[self.ctrl_idxs]

            # Check that the 
            if np.sum(np.hstack([self.ctrlr_dof_xyz, self.ctrlr_dof_abg])) > len(self.joint_ids):
                print("Fewer DOF than specified")
        
        # Initialize dicts to keep track of the state variables and locks
        self.__state_var_map: Dict[DeviceState, function] = {
            DeviceState.Q : lambda device: self.sim_interface.get_feedback(device)["q"],
            DeviceState.Q_ACTUATED : lambda device: self.sim_interface.get_feedback(device)["q"],
            DeviceState.DQ : lambda device: self.sim_interface.get_feedback(device)["dq"],
            DeviceState.DQ_ACTUATED : lambda device: self.sim_interface.get_feedback(device)["dq"],
            DeviceState.DDQ : lambda device: self.sim_interface.get_feedback(device)["dqq"],
            DeviceState.EE_XYZ : lambda device: self.sim_interface.get_xyz(device.EE),
            DeviceState.EE_XYZ_VEL : lambda device: self.sim_interface.get_xvelp(device, device.EE),
            DeviceState.EE_QUAT : lambda device: self.sim_interface.get_orientation(device.EE),
            DeviceState.FORCE : lambda device: self.__get_force(),
            DeviceState.TORQUE : lambda device: self.__get_torque(),
            DeviceState.J : lambda device: self.__get_jacobian()
        }
        
        self.__state: Dict[DeviceState, Any] = dict()
        self.__state_locks: Dict[DeviceState, Lock] = dict([(key, Lock()) for key in DeviceState])
        
        # These are the that keys we should use when returning data from get_all_states()
        self.concise_state_vars = [
            DeviceState.Q_ACTUATED, 
            DeviceState.DQ_ACTUATED, 
            DeviceState.EE_XYZ, 
            DeviceState.EE_XYZ_VEL, 
            DeviceState.EE_QUAT,
            DeviceState.FORCE,
            DeviceState.TORQUE
        ]

    def __get_jacobian(self, full=False):
        """
        NOTE: Returns either:
        1) The full jacobian (of the Device, using its EE), if full==True 
        2) The full jacobian evaluated at the controlled DoF, if full==False 
        The parameter, full=False, is added in case we decide for the get methods 
        to take in arguments (currently not supported).
        """
        J = self.J(self.EE)
        if full == False:
            J = J[self.ctrlr_dof]
        return J

    def __get_R(self, as_quat=True):
        """
        Get rotation matrix for device's ft_frame
        """
        if self.name == "ur5right":
            return self.sim_interface.get_orientation("ft_frame_ur5right", as_quat, 'site')
        if self.name == "ur5left":
            return self.sim_interface.get_orientation("ft_frame_ur5left", as_quat, 'site')

    def __get_force(self):
        """
        Get the external forces, used (for admittance control) acting upon
        the gripper sensors
        """
        sensor_data = self.sim_interface.get_sensor_data()
        if self.name == "ur5right":
            force = np.matmul(self.__get_R(False), sensor_data[0:3])
            return force
        if self.name == "ur5left":
            force = np.matmul(self.__get_R(False), sensor_data[6:9])
            return force
        else:
            return np.zeros(3)
            
    def __get_torque(self):
        """
        Get the external torques, used (for admittance control) acting upon
        the gripper sensors
        """
        sensor_data = self.sim_interface.get_sensor_data()
        if self.name == "ur5right":
            force = np.matmul(self.__get_R(False), sensor_data[3:6])
            return force
        if self.name == "ur5left":
            force = np.matmul(self.__get_R(False), sensor_data[9:12])
            return force
        else:
            return np.zeros(3)

    def __set_state(self, state_var: DeviceState):
        """
        Set the state of the device corresponding to the key value (if exists)    
        """
        assert not self.use_sim_state
        self.__state_locks[state_var].acquire()
        var_func = self.__state_var_map[state_var]
        var_value = var_func(self)
        self.__state[state_var] = copy.copy(var_value) # Make sure to copy (or else reference will stick to Dict value)
        self.__state_locks[state_var].release()

    def get_state(self, state_var: DeviceState):
        """
        Get the state of the device corresponding to the key value (if exists)
        """
        if self.use_sim_state:
            func = self.__state_var_map[state_var]
            state = copy.copy(func(self))
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state
    
    def get_all_states(self):
        return dict([(key, self.get_state(key)) for key in self.concise_state_vars])
    
    def update_state(self):
        """
        This should running in a thread: Robot.start()
        """
        assert not self.use_sim_state
        for var in DeviceState:
            self.__set_state(var)
        
    def get_all_joint_ids(self):
        return self.joint_ids_all
    
    def get_actuator_joint_ids(self):
        return self.joint_ids
    
    def get_gripper_joint_ids(self):
        return self.gripper_ids