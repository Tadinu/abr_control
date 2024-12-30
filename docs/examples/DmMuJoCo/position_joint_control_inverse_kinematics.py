"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import sys
import numpy as np

from abr_control.controllers import path_planners
from abr_control.utils import transformations
from abr_control.app.mujoco_app import MujocoApp

class PositionJointIKControl(MujocoApp):
    def tick(self, update_display = True):
        global n_timesteps, count, robot_model, path_planner, ee_id, target_id

        # get arm feedback
        feedback = self.get_feedback(robot_model)

        if count % n_timesteps == 0:
            target_xyz = np.array(
                [
                    np.random.random() * 0.5 - 0.25,
                    np.random.random() * 0.5 - 0.25,
                    np.random.random() * 0.5 + 0.5,
                ]
            )
            # update the position of the target
            self.set_mocap_xyz_by_id(target_id, target_xyz)

            # can use 3 different methods to calculate inverse kinematics
            # see inverse_kinematics.py file for details
            target_quat = self.get_mocap_orientation_by_id(target_id)
            #target_euler_angles = transformations.euler_from_quaternion(target_quat, "rxyz")
            #target_rot_mat = self.get_rotation_matrix_by_id(target_id).reshape(3, 3)
            #target_orientation = transformations.euler_from_matrix(target_rot_mat, "sxyz")
            path_planner.generate_path(
                position=feedback["q"],
                target_position=target_xyz,
                target_quat=target_quat,
                method=3,
                dt=0.005,
                n_timesteps=n_timesteps,
                plot=False,
            )

        # returns desired [position, velocity]
        target = path_planner.next()[0]

        # use position control
        #print("target angles: ", target[: robot_model.N_JOINTS])
        self.send_target_angles(robot_model, target[: robot_model.N_JOINTS])

        count += 1
        return super().tick(update_display)

# Main entrypoint
if __name__ == "__main__":
    # [arm_model]
    if len(sys.argv) > 1:
        arm_model = sys.argv[1]
    else:
        arm_model = "ur5"

    # [arm_name]: must match one defined in [default_{arm_model}.yaml]
    arm_name = f'{arm_model}_0'

    # create the Mujoco sim app and connect up
    OFFSCREEN_RENDERING = True
    sim_app = PositionJointIKControl(app_config_file=f"arms/{arm_model}/configs/default_{arm_model}.yaml",
                                     scene_xml=f"arms/{arm_model}/{arm_model}.xml",
                                     dt=0.001, visualize=True,
                                     create_offscreen_rendercontext=OFFSCREEN_RENDERING)

    # initialize [arm_model]-specific configs
    robot = sim_app.get_robot(arm_name)
    assert robot, f'Make sure [{arm_name}] is specified as robot_name in {sim_app.app_config_path}'
    robot_model = robot.get_device_model(arm_model)
    print(f'Robot [{robot.name}] - model[{robot_model.name}]')

    ee_id = sim_app.sim_model.name2id('EE', 'body')
    target_id = sim_app.sim_model.name2id('target_orientation', 'body')

    # start pose
    sim_app.send_target_angles(robot_model, robot_model.start_angles)

    # path planner
    path_planner = path_planners.InverseKinematics(robot_model)

    # Open main window
    count = 0
    n_timesteps = 2000
    try:
        sim_app.run(arm_name, device_model_name_list=[arm_model])
    finally:
        print(f"App {sim_app.scene_xml_path} ended")
