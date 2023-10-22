from .device_model import DeviceModel, DeviceState
import numpy as np
import mujoco as mjp
import time
from enum import Enum
from threading import Lock
from typing import Dict, Any, List
import copy

class RobotState(Enum):
    M = 'INERTIA'
    DQ = 'DQ'
    J = 'JACOBIAN'

class BaseRobot():
    def __init__(self, robot_name, sim, device_models: List[DeviceModel], use_sim_state, collect_hz=1000):
        self.sim = sim
        self.__use_sim = use_sim_state
        self.device_models = device_models
        self.device_models_dict: Dict[str, DeviceModel] = dict()
        for dev in self.device_models:
            self.device_models_dict[dev.name] = dev

        self.name = robot_name
        self.num_scene_joints = self.sim.model.ptr.nv
        self.M_vec = np.zeros((self.num_scene_joints, self.num_scene_joints))
        self.joint_ids_all = np.array([], dtype=np.int32)
        for dev in self.device_models:
            self.joint_ids_all = np.hstack([self.joint_ids_all, dev.joint_ids_all])
        self.joint_ids_all = np.sort(np.unique(self.joint_ids_all))
        self.num_joints_total = len(self.joint_ids_all)
        self.running = False
        self.__state_locks: Dict[RobotState, Lock] = dict([(key, Lock()) for key in RobotState])
        self.__state_var_map: Dict[RobotState, function] = {
            RobotState.M : lambda : self.__get_M(),
            RobotState.DQ : lambda : self.__get_dq(),
            RobotState.J : lambda : self.__get_jacobian()
        }
        self.__state: Dict[RobotState, Any] = dict()
        self.data_collect_hz = collect_hz

    
    def __get_jacobian(self):
        """
            Return the Jacobians for all of the devices,
            so that OSC can stack them according to provided the target entries
        """
        Js = dict()
        J_idxs = dict()
        start_idx = 0
        for name, device_model in self.device_models_dict.items():
            J_sub = device_model.get_state(DeviceState.J)
            J_idxs[name] = np.arange(start_idx, start_idx + J_sub.shape[0])
            start_idx += J_sub.shape[0]
            J_sub = J_sub[:, self.joint_ids_all]
            Js[name] = J_sub
        return Js, J_idxs
    
    def __get_dq(self):
        #dq = self.sim.data.qvel[self.joint_ids_all]
        dq = np.zeros(self.joint_ids_all.shape)
        for dev in self.device_models:
            dq[dev.get_all_joint_ids()] = dev.get_state(DeviceState.DQ)
        return dq

    def __get_M(self):
        mjp.mj_fullM(self.sim.model.ptr, self.M_vec, self.sim.data.ptr.qM)
        M = self.M_vec.reshape(self.num_scene_joints, self.num_scene_joints)
        M = M[np.ix_(self.joint_ids_all, self.joint_ids_all)]
        return M

    def get_state(self, state_var: RobotState):
        if self.__use_sim:
            func = self.__state_var_map[state_var]
            state = copy.copy(func())
        else:
            self.__state_locks[state_var].acquire()
            state = copy.copy(self.__state[state_var])
            self.__state_locks[state_var].release()
        return state

    def __set_state(self, state_var: RobotState):
        assert self.__use_sim is False
        self.__state_locks[state_var].acquire()
        func = self.__state_var_map[state_var]
        value = func()
        self.__state[state_var] = copy.copy(value) # Make sure to copy (or else reference will stick to Dict value)
        self.__state_locks[state_var].release()

    def is_running(self):
        return self.running
    
    def is_using_sim(self):
        return self.__use_sim

    def __update_state(self):
        assert self.__use_sim is False
        for var in RobotState:
            self.__set_state(var)
    
    def start(self):
        assert self.running is False and self.__use_sim is False
        self.running = True
        interval = float(1.0/float(self.data_collect_hz))
        prev_time = time.time()
        while self.running:
            for dev in self.device_models:
                dev.update_state()
            self.__update_state()
            curr_time = time.time()
            diff = curr_time - prev_time
            delay = max(interval - diff, 0)
            time.sleep(delay)
            prev_time = curr_time
    
    def stop(self):
        assert self.running is True and self.__use_sim is False
        self.running = False

    def get_device_model(self, device_model_name: str) -> DeviceModel:
        return self.device_models_dict[device_model_name]

    def get_all_states(self):
        """
        Get's the state of all the devices connected plus the robot states
        """
        state = {}
        for device_model_name, device_model in self.device_models_dict.items():
            state[device_model_name] = device_model.get_all_states()
        
        for key in RobotState:
            state[key] = self.get_state(key)
        
        return state
    
    def get_device_states(self):
        """
        Get's the state of all the devices connected
        """
        state = {}
        for device_model_name, device_model in self.device_models_dict.items():
            state[device_model_name] = device_model.get_all_states()
        return state