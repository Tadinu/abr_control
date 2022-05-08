"""
Running operational space control using Mujoco. The controller will
move the end-effector to the target object's position and orientation.

This example controls all 6 degrees of freedom (position and orientation),
and applies a second order path planner to both position and orientation targets

After termination the script will plot results
"""
import sys
import numpy as np

from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.controllers import OSC, Damping
from abr_control.controllers.path_planners import PathPlanner
from abr_control.controllers.path_planners.position_profiles import Linear
from abr_control.controllers.path_planners.velocity_profiles import Gaussian
from abr_control.interfaces.mujoco import AbrMujoco
from abr_control.utils import transformations

from main_window import MainWindow

# initialize our robot config
if len(sys.argv) > 1:
    arm_model = sys.argv[1]
else:
    arm_model = "jaco2"
# initialize our robot config for the jaco2
robot_config = arm(arm_model)

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
    kp=30,  # position gain
    kv=20,
    ko=180,  # orientation gain
    null_controllers=[damping],
    vmax=None,  # [m/s, rad/s]
    # control all DOF [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, True, True, True, True],
)

feedback = sim_interface.get_feedback()
hand_xyz = robot_config.Tx("EE", feedback["q"])

path_planner = PathPlanner(
    pos_profile=Linear(), vel_profile=Gaussian(dt=DT, acceleration=2)
)


# set up lists for tracking data
ee_track = []
ee_angles_track = []
target_track = []
target_angles_track = []
first_pass = True

ee_id = sim_interface.model.name2id('EE', 'body')
target_orientation_id = sim_interface.sim.model.name2id('target_orientation', 'body')
path_planner_orientation_id = sim_interface.sim.model.name2id('path_planner_orientation', 'body')

count = 0
def tick():
    global sim_interface, robot_config, ctrlr, path_planner, ee_track, ee_angles_track, target_track, target_angles_track, first_pass, count
    global ee_id, target_orientation_id, path_planner_orientation_id

    # get arm feedback
    feedback = sim_interface.get_feedback()
    hand_xyz = sim_interface.get_xyz_by_id(ee_id)
    if first_pass or count == path_planner.n_timesteps + 500:
        count = 0
        first_pass = False

        # pregenerate our path and orientation planners
        q = sim_interface.get_orientation_by_id(ee_id)
        starting_orientation = transformations.euler_from_quaternion(q, axes="rxyz")

        mag = 0.6
        target_position = np.random.random(3) * 0.5
        target_position = target_position / np.linalg.norm(target_position) * mag

        target_orientation = np.random.uniform(low=-np.pi, high=np.pi, size=3)

        path_planner.generate_path(
            start_position=hand_xyz,
            target_position=target_position,
            start_orientation=starting_orientation,
            target_orientation=target_orientation,
            max_velocity=2,
        )

        sim_interface.set_mocap_xyz_by_id(target_orientation_id, target_position)
        sim_interface.set_mocap_orientation_by_id(
            target_orientation_id,
            transformations.quaternion_from_euler(
                target_orientation[0],
                target_orientation[1],
                target_orientation[2],
                "rxyz",
            ),
        )

    next_target = path_planner.next()
    pos = next_target[:3]
    vel = next_target[3:6]
    orient = next_target[6:9]
    target = np.hstack([pos, orient])

    sim_interface.set_mocap_xyz_by_id(path_planner_orientation_id, target[:3])
    sim_interface.set_mocap_orientation_by_id(
        path_planner_orientation_id,
        transformations.quaternion_from_euler(
            orient[0], orient[1], orient[2], "rxyz"
        ),
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
    ee_matrix = sim_interface.get_rotation_matrix_by_id(ee_id).reshape(3,3)
    ee_angles_track.append(
        transformations.euler_from_matrix(
            ee_matrix, axes="rxyz"
        )
    )
    target_track.append(np.copy(target[:3]))
    target_angles_track.append(np.copy(target[3:]))
    count += 1
    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_config)
    main_window.exec(tick)
finally:
    ee_track = np.array(ee_track).T
    ee_angles_track = np.array(ee_angles_track).T
    target_track = np.array(target_track).T
    target_angles_track = np.array(target_angles_track).T

    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

        label_pos = ["x", "y", "z"]
        label_or = ["a", "b", "g"]
        c = ["r", "g", "b"]

        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(311)
        ax1.set_ylabel("3D position (m)")
        for ii, ee in enumerate(ee_track):
            ax1.plot(ee, label=f"EE: {label_pos[ii]}", c=c[ii])
            ax1.plot(
                target_track[ii],
                label=f"Target: {label_pos[ii]}",
                c=c[ii],
                linestyle="--",
            )
        ax1.legend()

        ax2 = fig.add_subplot(312)
        for ii, ee in enumerate(ee_angles_track):
            ax2.plot(ee, label=f"EE: {label_or[ii]}", c=c[ii])
            ax2.plot(
                target_angles_track[ii],
                label=f"Target: {label_or[ii]}",
                c=c[ii],
                linestyle="--",
            )
        ax2.set_ylabel("3D orientation (rad)")
        ax2.set_xlabel("Time (s)")
        ax2.legend()

        ee_track = ee_track.T
        target_track = target_track.T
        ax3 = fig.add_subplot(313, projection="3d")
        ax3.set_title("End-Effector Trajectory")
        ax3.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label="ee_xyz")
        ax3.plot(
            target_track[:, 0],
            target_track[:, 1],
            target_track[:, 2],
            label="ee_xyz",
            c="g",
            linestyle="--",
        )
        ax3.scatter(
            target_track[-1, 0],
            target_track[-1, 1],
            target_track[-1, 2],
            label="target",
            c="g",
        )
        ax3.legend()
        plt.show()
