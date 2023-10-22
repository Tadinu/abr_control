"""
Move the jao2 Mujoco arm to a target position.
The simulation ends after 1500 time steps, and the
trajectory of the end-effector is plotted in 3D.
"""
import numpy as np

from abr_control.arms.mujoco_model import MujocoModel as arm
from abr_control.controllers import Joint
from abr_control.interfaces.abr_mujoco import AbrMujoco
from abr_control.utils import transformations
from abr_control.utils.transformations import quaternion_conjugate, quaternion_multiply

from abr_control.app.main_window import MainWindow

# initialize our robot config for the jaco2
robot_model = arm("mujoco_balljoint.xml", folder=".", use_sim_state=False)

# create the Mujoco sim_interface & connect
OFFSCREEN_RENDERING=True
DT = 0.001
sim_interface = AbrMujoco(robot_model, dt=DT, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
# Connect to Mujoco instance, creating sim_interface viewer's main window
sim_interface.connect()
sim_interface.init_viewer()
#sim_interface.send_target_angles(robot_model.START_ANGLES)

# instantiate controller
kp = 100
kv = np.sqrt(kp)
ctrlr = Joint(
    robot_model,
    kp=kp,
    kv=kv,
    quaternions=[True],
)

# set up lists for tracking data
q_track = []
target_track = []
error_track = []

ee_id = sim_interface.sim.model.name2id('EE', 'body')
target_id = sim_interface.sim.model.name2id('target', 'body')
target_geom_id = sim_interface.sim.model.name2id('target', 'geom')
green = [0, 0.9, 0, 0.5]
red = [0.9, 0, 0, 0.5]
threshold = 0.002  # threshold distance for being within target before moving on

# get the end-effector's initial position
def set_mocap_target():
    global sim_interface, target_id, ee_id
    sim_interface.set_mocap_xyz_by_id(target_id, xyz=sim_interface.get_xyz_by_id(ee_id))

np.random.seed(0)
target_quaternions = [
    transformations.unit_vector(transformations.random_quaternion()) for ii in range(4)
]
target_index = 0
target = target_quaternions[target_index]
set_mocap_target()

count = 0
def tick():
    global sim_interface, robot_model, ctrlr, q_track, target, target_index, target_track, error_track, target_geom_id, target_quaternions, count
    global target_id, ee_id

    # get joint angle and velocity feedback
    feedback = sim_interface.get_feedback()

    # calculate the control signal
    u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=target,
    )

    # send forces into Mujoco, step the sim forward
    sim_interface.send_forces(u)

    # track data
    q_track.append(np.copy(feedback["q"]))
    target_track.append(np.copy(target))

    # calculate the distance between quaternions
    error = quaternion_multiply(
        target,
        quaternion_conjugate(feedback["q"]),
    )
    error = 2 * np.arctan2(np.linalg.norm(error[1:]) * -np.sign(error[0]), error[0])
    # quaternion distance for same angles can be 0 or 2*pi, so use a sine
    # wave here so 0 and 2*np.pi = 0
    error = np.sin(error / 2)
    error_track.append(np.copy(error))

    if abs(error) < threshold:
        sim_interface.sim.model.geom_rgba[target_geom_id] = green
        count += 1
    else:
        count = 0
        sim_interface.sim.model.geom_rgba[target_geom_id] = red

    if count >= 1000:
        target_index += 1
        target = target_quaternions[target_index % len(target_quaternions)]
        set_mocap_target()
        count = 0

    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_model)
    main_window.exec(tick)
finally:
    q_track = np.array(q_track)
    target_track = np.array(target_track)

    if q_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

        plt.figure(figsize=(8, 6))
        plt.ylabel("Distance (m)")
        plt.xlabel("Time (ms)")
        plt.title("Distance to target")
        plt.plot(np.array(error_track))
        plt.show()
