from typing import Dict
import time
import os
import yaml

import numpy as np
import mujoco as mjp

#import dm_control.mujoco as dm_mujoco
import abr_control
from abr_control.interfaces import AbrMujoco
from abr_control.arms import DeviceModel
from abr_control.arms import BaseRobot
from .main_window import MainWindow

class MujocoApp(AbrMujoco):
    def __init__(self, app_config_file : str = None, scene_xml : str = None, use_sim_state : bool = True,
                 dt=0.001, visualize=True,
                 create_offscreen_rendercontext=False):
        # Connect to Mujoco here-in
        super().__init__(scene_xml=scene_xml, dt=dt, visualize=visualize,
                         create_offscreen_rendercontext=create_offscreen_rendercontext)    

        # Create [self.config[]]
        self.main_dir = os.path.dirname(abr_control.__file__)
        self.app_config_path = os.path.join(self.main_dir, app_config_file)
        with open(self.app_config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # 1- [self.config[]] -> [self.device_models]
        dev_configs = self.config['devices']
        self.init_device_models([DeviceModel(self, device_yml=dev_cfg, xml_file=self.scene_xml_path if len(dev_configs) == 1 else None, use_sim_state=use_sim_state) for dev_cfg in dev_configs])
        self.devices = np.array(self.device_models)

        # 2- [Device joints] from ee_names ('ur_stand_dummy' is inactive)
        devices_dict = dict([(dev.name, dev) for dev in self.device_models])
        devices_ee_dict = dict([(dev, dev.EE) for _, dev in devices_dict.items()])
        self.init_device_joints(devices_ee_dict)

        # 3- device start pose, requiring [Device joints]
        for device_model in self.device_models:
            self.init_device_start_pose(device_model)

        # 4- init single device_model
        for device_model in self.device_models:
            device_model.calc_joints_data()

        # 3- Create [self.robots]
        self.create_robots(self.config['robots'], use_sim_state)
        self.controller_configs = self.config['controller_configs']

        self.timer_running = False

    def create_robots(self, robot_yml: Dict, use_sim_state: bool):
        self.robots = np.array([])
        all_robot_device_idxs = np.array([], dtype=np.int32)
        for robot_cfg in robot_yml:
            robot_device_idxs = robot_cfg['device_ids']
            all_robot_device_idxs = np.hstack([all_robot_device_idxs, robot_device_idxs])
            robot = BaseRobot(robot_cfg['name'], self.sim, self.devices[robot_device_idxs], use_sim_state)
            self.robots = np.append(self.robots, robot)
        
        all_idxs = np.arange(len(self.devices))
        keep_idxs = np.setdiff1d(all_idxs, all_robot_device_idxs)
        self.devices = np.hstack([self.devices[keep_idxs], self.robots])
    
    def sleep_for(self, sleep_time: float):
        assert self.timer_running == False
        self.timer_running = True
        time.sleep(sleep_time)
        self.timer_running = False

    def get_robot(self, robot_name: str) -> BaseRobot:
        for robot in self.robots:
            if robot.name == robot_name:
                return robot
        return None

    def get_controller_config(self, name: str) -> Dict:
        ctrlr_conf = self.config['controller_configs']
        for entry in ctrlr_conf:
            if entry['name'] == name:
                return entry
    
    def set_free_joint_qpos(self, free_joint_name, quat=None, pos=None):
        jnt_id = self.sim.model.name2id(free_joint_name, 'joint')
        offset = self.sim_model.jnt_qposadr[jnt_id]
        if quat is not None:
            quat_idxs = np.arange(offset+3, offset+7) # Grab the quaternion idxs
            self.data_ptr.qpos[quat_idxs] = quat
        if pos is not None:
            pos_idxs = np.arange(offset, offset+3)
            self.data_ptr.qpos[pos_idxs] = pos

    def run(self, robot_name, device_model_name_list, randomize = False):
        # Exec main window
        main_window = MainWindow(self, [self.get_robot(robot_name).get_device_model(device_model_name) for device_model_name in device_model_name_list])
        main_window.exec(self.tick)