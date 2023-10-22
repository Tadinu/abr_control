"""
A basic script for connecting and moving the arm to a target
configuration in joint space, offset from its starting position.
The simulation simulates 2500 time steps and then plots the results.
"""
import sys
import numpy as np

from abr_control.arms.mujoco_model import MujocoModel as arm
from abr_control.controllers import Joint
from abr_control.interfaces.abr_mujoco import AbrMujoco

from abr_control.app.main_window import MainWindow

if len(sys.argv) > 1:
    arm_model = sys.argv[1]
else:
    arm_model = "jaco2"
# initialize our robot config for the jaco2
robot_model = arm(arm_model)

# create sim_interface and connect
DT = 0.001
OFFSCREEN_RENDERING=True
sim_interface = AbrMujoco(robot_model=robot_model, dt=DT, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
sim_interface.connect()
sim_interface.init_viewer()
sim_interface.send_target_angles(robot_model.START_ANGLES)

# instantiate the REACH controller for the jaco2 robot
ctrlr = Joint(robot_model, kp=20, kv=10)

# make the target an offset of the current configuration
feedback = sim_interface.get_feedback()
target = feedback["q"] + np.random.random(robot_model.N_JOINTS) * 2 - 1
print(f'Target {target}')
# set up arrays for tracking end-effector and target position
q_track = []

count = 0
def tick():
    global sim_interface, robot_model, ctrlr, count, q_track, target, count
    if count>=2500:
        return sim_interface.tick()

    # get joint angle and velocity feedback
    feedback = sim_interface.get_feedback()

    # calculate the control signal
    u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=target,
    )

    # add gripper forces
    u = np.hstack((u, np.zeros(robot_model.N_GRIPPER_JOINTS)))

    # send forces into Mujoco, step the sim forward
    sim_interface.send_forces(u)

    # track joint angles
    #print(f'count {count} q {feedback["q"]}')
    q_track.append(np.copy(feedback["q"]))
    count += 1
    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_model)
    main_window.exec(tick)
finally:
    q_track = np.array(q_track)
    if q_track.shape[0] > 0:
        import matplotlib.pyplot as plt

        plt.plot((q_track + np.pi) % (np.pi * 2) - np.pi)
        plt.gca().set_prop_cycle(None)
        plt.plot(
            np.ones(q_track.shape) * ((target + np.pi) % (np.pi * 2) - np.pi), "--"
        )
        plt.legend(range(robot_model.N_JOINTS))
        plt.tight_layout()
        plt.show()
