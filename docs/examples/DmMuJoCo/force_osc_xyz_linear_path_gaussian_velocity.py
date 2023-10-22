"""
Running operational space control using Mujoco. The controller will
move the end-effector to the target object's X position and orientation.

The cartesian direction being controlled is set in the first three booleans
of the ctrlr_dof parameter
"""
import sys
import numpy as np

from abr_control.arms.mujoco_model import MujocoModel as arm
from abr_control.controllers import OSC, Damping
from abr_control.controllers.path_planners import PathPlanner
from abr_control.controllers.path_planners.position_profiles import Linear
from abr_control.controllers.path_planners.velocity_profiles import Gaussian
from abr_control.interfaces.abr_mujoco import AbrMujoco

from abr_control.app.main_window import MainWindow

max_a = 2

if len(sys.argv) > 1:
    arm_model = sys.argv[1]
else:
    arm_model = "jaco2"
# initialize our robot config for the jaco2
robot_model = arm(arm_model)

ctrlr_dof = [True, True, True, False, False, False]
dof_labels = ["x", "y", "z", "a", "b", "g"]
dof_print = f"* DOF Controlled: {np.array(dof_labels)[ctrlr_dof]} *"
stars = "*" * len(dof_print)
print(stars)
print(dof_print)
print(stars)

# create the Mujoco sim_interface & connect
OFFSCREEN_RENDERING=True
DT = 0.001
sim_interface = AbrMujoco(robot_model, dt=DT, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
# Connect to Mujoco instance, creating sim_interface viewer's main window
sim_interface.connect()
sim_interface.init_viewer()
sim_interface.send_target_angles(robot_model.START_ANGLES)

# damp the movements of the arm
damping = Damping(robot_model, kv=10)
# create opreational space controller
ctrlr = OSC(
    robot_model,
    kp=30,
    kv=20,
    ko=180,
    null_controllers=[damping],
    vmax=[10, 10],  # [m/s, rad/s]
    # control (x, alpha, beta, gamma) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=ctrlr_dof,
)

path_planner = PathPlanner(
    pos_profile=Linear(), vel_profile=Gaussian(dt=DT, acceleration=max_a)
)

# set up lists for tracking data
ee_track = []
target_track = []

ee_id = sim_interface.model.name2id('EE', 'body')
target_id = sim_interface.model.name2id('target', 'body')
target_orientation_id = sim_interface.model.name2id('target_orientation', 'body')
        
count = 0
at_target = 0
pos_target = None
def fetch_next_target():
    global sim_interface, path_planner, at_target
    next_target = np.array(
        [
            np.random.uniform(low=-0.4, high=0.4),
            np.random.uniform(low=-0.4, high=0.4),
            np.random.uniform(low=0.3, high=0.6),
        ]
    )

    path_planner.generate_path(
        start_position=sim_interface.get_xyz_by_id(ee_id), target_position=next_target, max_velocity=2
    )

    sim_interface.set_mocap_xyz_by_id(target_id, next_target)
    at_target = 0
    return next_target

def tick():
    global sim_interface, robot_model, ctrlr, path_planner, pos_target, count, at_target, ee_id, target_id, target_orientation_id

    if at_target >= 500:
        return sim_interface.tick()
    elif count > 5000:
        pos_target = fetch_next_target()
    
    filtered_target = path_planner.next()
    sim_interface.set_mocap_xyz_by_id(target_orientation_id, filtered_target[:3])

    feedback = sim_interface.get_feedback()
    hand_xyz = sim_interface.get_xyz_by_id(ee_id)

    u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=filtered_target,
    )

    # add gripper forces
    u = np.hstack((u, np.zeros(robot_model.N_GRIPPER_JOINTS)))

    # apply the control signal, step the sim forward
    sim_interface.send_forces(u)

    # track data
    hand_xyz = sim_interface.get_xyz_by_id(ee_id)
    ee_track.append(np.copy(hand_xyz))
    target_track.append(np.copy(pos_target[:3]))

    if np.linalg.norm(hand_xyz - pos_target) < 0.02:
        at_target += 1
    else:
        at_target = 0
    count += 1
    return sim_interface.tick()

# Initial target
pos_target = fetch_next_target()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_model)
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
        ax1.set_ylabel("3D position (m)")
        for ii, controlled_dof in enumerate(ctrlr_dof[:3]):
            if controlled_dof:
                ax1.plot(ee_track[:, ii], label=dof_labels[ii])
                ax1.plot(target_track[:, ii], "--")
        ax1.legend()

        ax3 = fig.add_subplot(212, projection="3d")
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
