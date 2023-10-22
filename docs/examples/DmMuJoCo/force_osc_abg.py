"""
Running operational space control using Mujoco. The controller will
move the end-effector to the target object's orientation.
"""
import sys
import numpy as np

from abr_control.controllers import OSC, Damping
from abr_control.app.mujoco_app import MujocoApp
from abr_control.utils import transformations

class Force_OSC_ABG(MujocoApp):
    def tick(self, update_display = True):
        global count, robot_model, ctrlr, rand_orient, ee_angles_track, target_angles_track, ee_id, target_id
        # print('count', count)
        if (count % 1000 == 0):
            # generate a random target orientation
            rand_orient = transformations.random_quaternion()
            print("New target orientation: ", rand_orient)

        # get arm feedback
        feedback = self.get_feedback(robot_model)
        hand_xyz = self.get_xyz_by_id(ee_id)

        # set our target to our ee_xyz since we are only focussing on orinetation
        self.set_mocap_xyz_by_id(target_id, hand_xyz)
        self.set_mocap_orientation_by_id(target_id, rand_orient)

        target_quat = self.get_mocap_orientation_by_id(target_id)
        target_euler_angles = transformations.euler_from_quaternion(target_quat, "rxyz")
        target = np.hstack(
            [
                self.get_xyz_by_id(target_id),
                target_euler_angles
            ]
        )

        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )

        #rc_matrix = robot_model.R("EE", feedback["q"])
        #rc_angles = transformations.euler_from_matrix(rc_matrix, axes="rxyz")

        # add gripper forces
        u_all = np.hstack((u, np.zeros(robot_model.N_GRIPPER_JOINTS)))

        # apply the control signal, step the sim forward
        use_joint_dyn_addrs = True
        self.send_forces(robot_model, u if use_joint_dyn_addrs else u_all, use_joint_dyn_addrs=use_joint_dyn_addrs)

        # track data
        ee_matrix = self.get_rotation_matrix_by_id(ee_id).reshape(3,3)
        ee_angles = transformations.euler_from_matrix(ee_matrix, axes="rxyz")

        ee_angles_track.append(ee_angles)
        target_angles_track.append(target_euler_angles)
        count += 1
        return super().tick(update_display)

# Main entrypoint
if __name__ == "__main__":
    # [arm_model]
    if len(sys.argv) > 1:
        arm_model = sys.argv[1]
    else:
        arm_model = "jaco2"

    # [arm_name]: must match one defined in [default_{arm_model}.yaml]
    arm_name = f'{arm_model}_0'

    # create the Mujoco sim app and connect up
    OFFSCREEN_RENDERING=True
    sim_app = Force_OSC_ABG(app_config_file=f"arms/{arm_model}/configs/default_{arm_model}.yaml",
                            scene_xml=f"arms/{arm_model}/{arm_model}.xml",
                            dt=0.001, visualize=True,
                            create_offscreen_rendercontext=OFFSCREEN_RENDERING)

    # initialize [arm_model]-specific configs
    robot = sim_app.get_robot(arm_name)
    assert robot, f'Make sure [{arm_name}] is specified as robot_name in {sim_app.app_config_path}'
    robot_model = robot.get_device_model(arm_model)
    print(f'Robot [{robot.name}] - model[{robot_model.name}]')

    # start pose
    sim_app.send_target_angles(robot_model, robot_model.start_angles)

    # damp the movements of the arm
    damping = Damping(robot_model, kv=10)
    # create operational space controller
    ctrlr = OSC(
        robot_model,
        kp=200,  # position gain
        ko=200,  # orientation gain
        null_controllers=[damping],
        # control (alpha, beta, gamma) out of [x, y, z, alpha, beta, gamma]
        ctrlr_dof=[False, False, False, True, True, True],
    )

    # set up lists for tracking data
    ee_angles_track = []
    target_angles_track = []

    ee_id = sim_app.sim_model.name2id('EE', 'body')
    target_id = sim_app.sim_model.name2id('target_orientation', 'body')

    count = 0
    rand_orient = transformations.random_quaternion()

    # Open main window
    try:
        sim_app.run(arm_name, device_model_name_list=[arm_model])
    finally:
        ee_angles_track = np.array(ee_angles_track)
        target_angles_track = np.array(target_angles_track)

        if ee_angles_track.shape[0] > 0:
            # plot distance from target and 3D trajectory
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(ee_angles_track)
            plt.gca().set_prop_cycle(None)
            plt.plot(target_angles_track, "--")
            plt.ylabel("3D orientation (rad)")
            plt.xlabel("Time (s)")
            plt.tight_layout()
            plt.show()