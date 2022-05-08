"""
Move the jao2 Mujoco arm to a target position.
The simulation ends after 1500 time steps, and the
trajectory of the end-effector is plotted in 3D.
"""
import os
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.controllers import OSC
from abr_control.interfaces.mujoco import AbrMujoco
from abr_control.utils import transformations

from main_window import MainWindow

# initialize our robot config for the jaco2
# os.path.abspath(os.getcwd())
robot_config = arm("mujoco_balljoint.xml", folder=os.path.dirname(os.path.abspath(__file__)))

# create the Mujoco sim_interface and connect up
OFFSCREEN_RENDERING=True
sim_interface = AbrMujoco(robot_config, dt=0.001, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
# Connect to Mujoco instance, creating sim_interface viewer's main window
sim_interface.connect()
sim_interface.init_viewer()

# instantiate controller
ctrlr = OSC(
    robot_config,
    kp=200,
    # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, True, False, False, False],
)

# set up lists for tracking data
ee_track = []
target_track = []

target_id = sim_interface.model.name2id('target', 'body')
target_geom_id = sim_interface.sim.model.name2id('target', 'geom')
green = [0, 0.9, 0, 0.5]
red = [0.9, 0, 0, 0.5]

# get the end-effector's initial position
targets = [
    np.array([-0.3, -0.3, 0.9]),
    np.array([0.3, -0.3, 0.9]),
    np.array([0.3, 0.3, 0.9]),
    np.array([-0.3, 0.3, 0.9]),
]
target_index = 0
sim_interface.set_mocap_xyz_by_id(target_id, xyz=targets[0])

ee_id = sim_interface.model.name2id('EE', 'body')

count = 0
def tick():
    global count, sim_interface, robot_config, ctrlr, ee_id, target_id, target_geom_id, targets
    # get joint angle and velocity feedback
    feedback = sim_interface.get_feedback()

    target = np.hstack(
        [
            sim_interface.get_xyz_by_id(target_id),
            transformations.euler_from_quaternion(
                sim_interface.get_orientation_by_id(target_id), "rxyz"
            ),
        ]
    )

    # calculate the control signal
    # robot_config.N_JOINTS = 3
    # inertia matrix in joint space
    #M = robot_config.M(np.hstack(feedback["q"]))
    #print(M, M.shape)
    u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=target,
    )

    # send forces into Mujoco, step the sim forward
    sim_interface.send_forces(u)

    # calculate end-effector position
    ee_xyz = sim_interface.get_xyx_by_id(ee_id)
    # track data
    ee_track.append(np.copy(ee_xyz))
    target_track.append(np.copy(target[:3]))

    error = np.linalg.norm(ee_xyz - target[:3])
    if error < 0.02:
        sim_interface.sim.model.geom_rgba[target_geom_id] = green
        count += 1
    else:
        count = 0
        sim_interface.sim.model.geom_rgba[target_geom_id] = red

    if count >= 50:
        target_index += 1
        sim_interface.set_mocap_xyz_by_id(
            target_id, xyz=targets[target_index % len(targets)]
        )
        count = 0
    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_config)
    main_window.exec(tick)
finally:
    ee_track = np.array(ee_track)
    target_track = np.array(target_track)

    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel("Distance (m)")
        ax1.set_xlabel("Time (ms)")
        ax1.set_title("Distance to target")
        ax1.plot(
            np.sqrt(np.sum((np.array(target_track) - np.array(ee_track)) ** 2, axis=1))
        )

        ax2 = fig.add_subplot(212, projection="3d")
        ax2.set_title("End-Effector Trajectory")
        ax2.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label="ee_xyz")
        ax2.scatter(
            target_track[0, 0],
            target_track[0, 1],
            target_track[0, 2],
            label="target",
            c="r",
        )
        ax2.legend()
        plt.show()