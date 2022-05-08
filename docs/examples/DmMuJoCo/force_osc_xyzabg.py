"""
Running operational space control using Mujoco. The controller will
move the end-effector to the target object's X position and orientation.

The cartesian direction being controlled is set in the first three booleans
of the ctrlr_dof parameter
"""
import sys
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.controllers import OSC, Damping
from abr_control.interfaces.mujoco import AbrMujoco
from abr_control.utils import transformations

from main_window import MainWindow

if len(sys.argv) > 1:
    arm_model = sys.argv[1]
else:
    arm_model = "jaco2"
# initialize our robot config for the jaco2
robot_config = arm(arm_model)

ctrlr_dof = [False, False, True, True, True, True]
dof_labels = ["x", "y", "z", "a", "b", "g"]
dof_print = f"* DOF Controlled: {np.array(dof_labels)[ctrlr_dof]} *"
stars = "*" * len(dof_print)
print(stars)
print(dof_print)
print(stars)

# create the Mujoco sim_interface & connect
OFFSCREEN_RENDERING=True
DT = 0.001
sim_interface = AbrMujoco(robot_config, dt=DT, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
# Connect to Mujoco instance, creating sim_interface viewer's main window
sim_interface.connect()
sim_interface.init_viewer()
sim_interface.send_target_angles(robot_config.START_ANGLES)

# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# create opreational space controller
ctrlr = OSC(
    robot_config,
    kp=30,
    kv=20,
    ko=180,
    null_controllers=[damping],
    vmax=[10, 10],  # [m/s, rad/s]
    # control (x, alpha, beta, gamma) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=ctrlr_dof,
)

target_xyz = np.array([0.3, 0.3, 0.5])
target_orientation = transformations.random_quaternion()

# set up lists for tracking data
ee_track = []
ee_angles_track = []
target_track = []
target_angles_track = []

ee_id = sim_interface.model.name2id('EE', 'body')
target_orientation_id = sim_interface.sim.model.name2id('target_orientation', 'body')

def tick():
    global sim_interface, robot_config, ctrlr, ee_track, ee_angles_track, target_track, target_angles_track, target_xyz, target_orientation
    global ee_id, target_orientation_id

    # get arm feedback
    feedback = sim_interface.get_feedback()
    hand_xyz = sim_interface.get_xyz_by_id(ee_id)

    for ii, dof in enumerate(ctrlr_dof[:3]):
        if not dof:
            target_xyz[ii] = hand_xyz[ii]

    sim_interface.set_mocap_xyz_by_id(target_orientation_id, target_xyz)
    sim_interface.set_mocap_orientation_by_id(target_orientation_id, target_orientation)
    target = np.hstack(
        [
            sim_interface.get_xyz_by_id(target_orientation_id),
            transformations.euler_from_quaternion(
                sim_interface.get_orientation_by_id(target_orientation_id), "rxyz"
            ),
        ]
    )

    u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=target,
    )

    # add gripper forces
    u = np.hstack((u, np.zeros(robot_config.N_GRIPPER_JOINTS)))

    # apply the control signal, step the sim forward
    sim_interface.send_forces(u)

    # track data
    hand_xyz = sim_interface.get_xyz_by_id(ee_id)
    ee_track.append(np.copy(hand_xyz))
    ee_angles_track.append(
        transformations.euler_from_matrix(
            hand_xyz, axes="rxyz"
        )
    )
    target_track.append(np.copy(target[:3]))
    target_angles_track.append(np.copy(target[3:]))
    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_config)
    main_window.exec(tick)
finally:
    ee_track = np.array(ee_track)
    ee_angles_track = np.array(ee_angles_track)
    target_track = np.array(target_track)
    target_angles_track = np.array(target_angles_track)

    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(311)
        ax1.set_ylabel("3D position (m)")
        for ii, controlled_dof in enumerate(ctrlr_dof[:3]):
            if controlled_dof:
                ax1.plot(ee_track[:, ii], label=dof_labels[ii])
                ax1.plot(target_track[:, ii], "--")
        ax1.legend()

        ax2 = fig.add_subplot(312)
        for ii, controlled_dof in enumerate(ctrlr_dof[3:]):
            if controlled_dof:
                ax2.plot(ee_angles_track[:, ii], label=dof_labels[ii + 3])
                ax2.plot(target_angles_track[:, ii], "--")
        ax2.set_ylabel("3D orientation (rad)")
        ax2.set_xlabel("Time (s)")
        ax2.legend()

        ax3 = fig.add_subplot(313, projection="3d")
        ax3.set_title("End-Effector Trajectory")
        ax3.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label="ee_xyz")
        ax3.scatter(
            target_track[0, 0],
            target_track[0, 1],
            target_track[0, 2],
            label="target",
            c="g",
        )
        ax3.legend()
        plt.show()
