"""
Example using the mujoco sim_interface with a three link arm
reaching with a path planner

To plot the joint angles call the script with True
passed through sys.argv
    python threelink.py True
"""

import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

from abr_control.arms.mujoco_model import MujocoModelfrom abr_control.controllers import OSC
from abr_control.controllers.path_planners import PathPlanner
from abr_control.controllers.path_planners.position_profiles import Linear
from abr_control.controllers.path_planners.velocity_profiles import Gaussian
from abr_control.interfaces.abr_mujoco import AbrMujoco

from abr_control.app.main_window import MainWindow

if len(sys.argv) > 1:
    use_wall_clock = True
else:
    use_wall_clock = False
    print(
        "To plan the path based on real time instead of steps, append"
        + " 'True' to your script call"
    )


if len(sys.argv) > 1:
    show_plot = sys.argv[1]
else:
    show_plot = False
model_filename = "threejoint"

robot_model = MujocoModel(model_filename)

# create the Mujoco sim_interface and connect
OFFSCREEN_RENDERING=True
DT=0.001
sim_interface = AbrMujoco(robot_model, dt=DT, visualize=True, create_offscreen_rendercontext=OFFSCREEN_RENDERING)
# Connect to Mujoco instance, creating sim_interface viewer's main window
sim_interface.connect()
sim_interface.init_viewer()
sim_interface.send_target_angles(robot_model.START_ANGLES)

ctrlr = OSC(
    robot_model, kp=30, kv=20, ctrlr_dof=[True, True, True, False, False, False]
)

sim_interface.send_target_angles(np.ones(3))

target_xyz = np.array([0.1, 0.1, 0.3])
sim_interface.set_mocap_xyz("target", target_xyz)

sim_interface.set_mocap_xyz("hand", np.array([0.2, 0.4, 1]))

target_geom_id = sim_interface.sim.model.name2id('target', 'geom')
green = [0, 0.9, 0, 0.5]
red = [0.9, 0, 0, 0.5]

# create our path planner
params = {}
time_elapsed = 0
if use_wall_clock:
    run_time = 4  # wall clock time to run each trajectory for
    params["n_timesteps"] = 100  # time steps each trajectory lasts
    time_elapsed = np.copy(run_time)
    count = 0
else:
    params["n_timesteps"] = 2000  # time steps each trajectory lasts
    count = np.copy(params["n_timesteps"])
    time_elapsed = 0.0
path_planner = PathPlanner(
    pos_profile=Linear(), vel_profile=Gaussian(dt=DT, acceleration=1)
)

ee_track = []
target_track = []

update_target = True
ee_name = 'EE'

ee_id = sim_interface.model.name2id(ee_name, 'body')
target_id = sim_interface.model.name2id('target', 'body')
path_planner_id = sim_interface.model.name2id('target', 'body')

count = 0
def tick():
    global sim_interface, robot_model, ctrlr, target_xyz, use_wall_clock, time_elapsed, update_target, ee_track, target_track, count
    global ee_id, target_id, path_planner_id, path_planner

    start = timeit.default_timer()
    # get arm feedback
    feedback = sim_interface.get_feedback()
    hand_xyz = sim_interface.get_xyz_by_id(ee_id)

    if update_target:
        count = 0
        time_elapsed = 0.0
        target_xyz[0] = np.random.uniform(0.2, 0.3) * np.sign(
            np.random.uniform(-1, 1)
        )
        target_xyz[1] = np.random.uniform(0.2, 0.25) * np.sign(
            np.random.uniform(-1, 1)
        )
        target_xyz[2] = np.random.uniform(0.3, 0.4)
        # update the position of the target
        sim_interface.set_mocap_xyz_by_id(target_id, target_xyz)

        generated_path = path_planner.generate_path(
            start_position=hand_xyz,
            target_position=target_xyz,
            max_velocity=1,
            plot=False,
        )
        pos_path = generated_path[:, :3]
        vel_path = generated_path[:, 3:6]

        if use_wall_clock:
            pos_path = path_planner.convert_to_time(
                path=pos_path, time_length=run_time
            )
            vel_path = path_planner.convert_to_time(
                path=vel_path, time_length=run_time
            )

    # get next target along trajectory
    if use_wall_clock:
        target = [function(min(time_elapsed, run_time)) for function in pos_path]
        target_velocity = [
            function(min(time_elapsed, run_time)) for function in vel_path
        ]
    else:
        next_target = path_planner.next()
        target = next_target[:3]
        target_velocity = next_target[3:]

    sim_interface.set_mocap_xyz_by_id(path_planner_id, target)
    # generate an operational space control signal
    u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=np.hstack((target, np.zeros(3))),
        ref_frame=ee_name,
    )

    # apply the control signal, step the sim forward
    sim_interface.send_forces(u)

    hand_xyz = sim_interface.get_xyz_by_id(ee_id)
    ee_track.append(np.copy(hand_xyz))
    target_track.append(hand_xyz)
    count += 1
    time_elapsed += timeit.default_timer() - start

    error = np.linalg.norm(hand_xyz - target_xyz)
    if error < 0.02:
        sim_interface.sim.model.geom_rgba[target_geom_id] = green
    else:
        sim_interface.sim.model.geom_rgba[target_geom_id] = red

    if count % 500 == 0:
        print("error: ", error)

    if use_wall_clock:
        # either update target every 1s
        update_target = time_elapsed >= path_planner.time_to_converge + 2
    else:
        # or update target when trajectory is done
        update_target = count == path_planner.n_timesteps + 500
    return sim_interface.tick()

# Open main window
try:
    main_window = MainWindow(sim_interface, robot_model)
    main_window.exec(tick)
finally:
    ee_track = np.array(ee_track)
    target_track = np.array(target_track)

    if show_plot:
        plt.figure()

        plt.subplot(1, 2, 1, projection="3d")
        plt.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2])
        plt.plot(
            target_track[:, 0],
            target_track[:, 1],
            target_track[:, 2],
            "rx",
            mew=3,
            lw=2,
        )
        plt.gca().set_xlim([-1, 1])
        plt.gca().set_ylim([-1, 1])
        plt.gca().set_zlim([0, 1])
        plt.gca().set_aspect("equal")

        plt.subplot(1, 2, 2)
        plt.plot(ee_track, lw=2)
        plt.gca().set_prop_cycle(None)
        plt.plot(target_track, "--", lw=2)

        plt.show()
