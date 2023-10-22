"""
A basic script for connecting to the arm and putting it in floating
mode, which only compensates for gravity. The end-effector position
is recorded and plotted when the script is exited (with ctrl-c).

In this example, the floating controller is applied in the joint space
"""
import sys
import numpy as np

from abr_control.arms.mujoco_model import MujocoModel as arm
from abr_control.controllers import Floating
from abr_control.interfaces.abr_mujoco import AbrMujoco

from abr_control.app.main_window import MainWindow

if len(sys.argv) > 1:
    arm_model = sys.argv[1]
else:
    arm_model = "jaco2"
# initialize our robot config
robot_model = arm(arm_model)

# create the Mujoco sim_interface & connect
OFFSCREEN_RENDERING=True
DT = 0.001
sim_interface = AbrMujoco(robot_model, dt=DT, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
# Connect to Mujoco instance, creating sim_interface viewer's main window
sim_interface.connect()
sim_interface.init_viewer()
sim_interface.send_target_angles(robot_model.START_ANGLES)

# instantiate the controller
ctrlr = Floating(robot_model, task_space=False, dynamic=True)

# set up arrays for tracking end-effector and target position
ee_track = []
q_track = []
      
ee_id = sim_interface.model.name2id('EE', 'body')

def tick():
    global sim_interface, robot_model, ctrlr, ee_track, q_track, ee_id
    # get joint angle and velocity feedback
    feedback = sim_interface.get_feedback()

    # calculate the control signal
    u = ctrlr.generate(q=feedback["q"], dq=feedback["dq"])

    # add gripper forces
    u = np.hstack((u, np.zeros(robot_model.N_GRIPPER_JOINTS)))

    # send forces into Mujoco
    sim_interface.send_forces(u)

    # get the position of the hand
    hand_xyz = sim_interface.get_xyz_from_id(ee_id, 'body')
    
    # track end effector position
    ee_track.append(np.copy(hand_xyz))
    q_track.append(np.copy(feedback["q"]))
    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_model)
    main_window.exec(tick)
finally:
    ee_track = np.array(ee_track)

    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(211)
        ax1.set_title("Joint Angles")
        ax1.set_ylabel("Angle (rad)")
        ax1.set_xlabel("Time (ms)")
        ax1.plot(q_track)
        ax1.legend()

        ax2 = fig.add_subplot(212, projection="3d")
        ax2.set_title("End-Effector Trajectory")
        ax2.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label="ee_xyz")
        ax2.legend()
        plt.show()