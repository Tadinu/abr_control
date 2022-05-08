"""
Minimal example using the mujoco sim_interface with a three link arm
reaching to a target with three degrees of freedom

To plot the joint angles call the script with True
passed through sys.argv
    python threelink.py True
"""
from lib2to3.pgen2.token import N_TOKENS
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.controllers import OSC
from abr_control.interfaces.mujoco import AbrMujoco

from main_window import MainWindow

print(
    "****************************************************************"
    + "***************************************"
)
print(
    "\n***Append 1, 2, or 3 to your function call to change between a"
    + " onejoint, twojoint, or threejoint arm***\n"
)
print(
    "****************************************************************"
    + "***************************************"
)
show_plot = False
if len(sys.argv) > 1:
    N_JOINTS = int(sys.argv[1])
else:
    N_JOINTS = 3

print('N_JOINTS', N_JOINTS)

if N_JOINTS == 1:
    model_filename = "onejoint"
    ctrlr_dof = [True, False, False, False, False, False]
elif N_JOINTS == 2:
    model_filename = "twojoint"
    ctrlr_dof = [True, True, False, False, False, False]
elif N_JOINTS == 3:
    model_filename = "threejoint"
    ctrlr_dof = [True, True, True, False, False, False]
else:
    raise Exception("Only 1-3 joint arms are available in this example")

robot_config = MujocoConfig(model_filename)

# create the Mujoco sim_interface and connect up
OFFSCREEN_RENDERING=True
DT = 0.005
sim_interface = AbrMujoco(robot_config, dt=DT, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
# Connect to Mujoco instance, creating sim_interface viewer's main window
sim_interface.connect()
sim_interface.init_viewer()
sim_interface.send_target_angles(robot_config.START_ANGLES)

ctrlr = OSC(robot_config, kp=10, kv=5, ctrlr_dof=ctrlr_dof)

sim_interface.send_target_angles(np.ones(N_JOINTS))

target = np.array([0.1, 0.3, 0.3, 0, 0, 0])
sim_interface.set_mocap_xyz("target", target[:3])
sim_interface.set_mocap_xyz("hand", np.array([0.2, 0.4, 1]))

q_track = []
ee_name = 'EE'
ee_id = sim_interface.model.name2id(ee_name, 'body')

def tick():
    global sim_interface, robot_config, ctrlr, q_track, ee_id, ee_name, target

    feedback = sim_interface.get_feedback()
    hand_xyz = sim_interface.get_xyz_by_id(ee_id)
    u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=target,
        ref_frame=ee_name,
    )
    #print(hand_xyz, feedback["q"], feedback["dq"])
    sim_interface.send_forces(u)

    if np.linalg.norm(hand_xyz[:N_JOINTS] - target[:N_JOINTS]) < 0.01:
        target[0] = np.random.uniform(0.2, 0.25) * np.sign(np.random.uniform(-1, 1))
        target[1] = np.random.uniform(0.2, 0.25) * np.sign(np.random.uniform(-1, 1))
        target[2] = np.random.uniform(0.4, 0.5)
        sim_interface.set_mocap_xyz("target", target[:3])

    q_track.append(feedback["q"])
    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_config)
    main_window.exec(tick)
finally:
    q_track = np.asarray(q_track)

    if show_plot:
        plt.figure(figsize=(30, 30))

        plt.plot(q_track)
        plt.ylabel("Joint Angles [rad]")
        plt.legend(["0", "1"])

        plt.show()
