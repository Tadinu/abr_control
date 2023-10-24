import os
from re import A
import mujoco as mjp
import dm_control.mujoco as dm_mujoco
from dm_control import _render as dm_render
from dm_control.viewer import gui as dm_view_gui
from dm_control.viewer import renderer as dm_view_renderer
#from dm_control.viewer import runtime as dm_view_runtime
#from dm_control.viewer import user_input as dm_view_user_input
from dm_control.viewer import util as dm_view_util
from dm_control.viewer import viewer as dm_viewer
from dm_control.viewer import views as dm_views
import numpy as np

import abr_control
from abr_control.utils import transformations

from .interface import Interface

MAX_FRONTBUFFER_SIZE = 2048
NULL_RENDERER = dm_view_renderer.NullRenderer()
class AbrMujoco(Interface):
    """An interface for MuJoCo using the mujoco package.

    Parameters
    ----------
    device_models: class instance
        contains all relevant model information about all robots in scene
        such as: number of joints, number of links, mass information etc.
    dt: float, optional (Default: 0.001)
        simulation time step in seconds
    visualize: boolean, optional (Default: True)
        turns visualization on or off
    create_offscreen_rendercontext: boolean, optional (Default: False)
        create the offscreen rendercontext behind the main visualizer
        (helpful for rendering images from other cameras without displaying them)
    """

    def __init__(
        self,
        scene_xml,
        dt=0.001,
        visualize=True,
        create_offscreen_rendercontext=False,
    ):
        super().__init__()
        self.main_dir = os.path.dirname(abr_control.__file__)
        self.visualize = visualize
        self.create_offscreen_rendercontext = create_offscreen_rendercontext
        self.time = 0 # accumulated time over steps

        # 1- connect to Mujoco, creating [self.sim/sim_model/model_ptr/data_ptr], instansiating [self.viewer] here-in
        self.connect(os.path.join(self.main_dir, scene_xml), dt)

        # 2- init viewer
        self.init_viewer()

    def init_device_models(self, device_models):
        """
        Parameters
        ----------
        device_models : class instance
            contains all relevant model information about all robots in scene
            such as: number of joints, number of links, mass information etc.
        """
        self.device_models = device_models
        for device_model in self.device_models:
            device_model.set_sim(self.sim)
            # set the time step for simulation
            device_model.sim_model.opt.timestep = self.dt

    def init_device_start_pose(self, device_model):
        # Start pose
        if device_model.name == "ur5right" or device_model.name == "ur5left" or device_model.name == "base":
            self.send_target_angles(device_model, device_model.start_angles)

    def connect(self, scene_xml_path, dt, joint_names=None, camera_id=-1, **kwargs):
        """
        joint_names: list, optional (Default: None)
            list of joint names to send control signal to and get feedback from
            if None, the joints in the kinematic tree connecting the end-effector
            to the world are used
        camera_id: int, optional (Default: -1)
            the id of the camera to use for the visualization
        """
        self.scene_xml_path = scene_xml_path
        self.sim = dm_mujoco.Physics.from_xml_path(scene_xml_path)
        self.sim.forward()  # run forward to fill in sim.data
        self.sim_model = self.sim.model # dm_control model
        self.model_ptr = self.sim_model.ptr # mujoco model
        self.data_ptr = self.sim.data.ptr # mujoco model data
        self.dt = dt  # time step
        self.joint_names = joint_names
        
        # if we want to use the offscreen render context create it before the
        # viewer so the corresponding window is behind the viewer
        if self.create_offscreen_rendercontext:
            self.offscreen = mjp.GLContext(max_width=MAX_FRONTBUFFER_SIZE, max_height=MAX_FRONTBUFFER_SIZE)
            self.offscreen.make_current()

        # create the visualizer
        if self.visualize:
            TITLE = 'MujocoViewer'
            WIDTH = 1024
            HEIGHT = 768
            self._viewport = dm_view_renderer.Viewport(WIDTH, HEIGHT)
            self._pause_subject = dm_view_util.ObservableFlag(True)
            self._time_multiplier = dm_view_util.TimeMultiplier(1.)
            self._frame_timer = dm_view_util.Timer()
            self.window = dm_view_gui.RenderWindow(WIDTH, HEIGHT, TITLE)
            self.viewer = dm_viewer.Viewer(self._viewport, self.window.mouse, self.window.keyboard)

        print("MuJoCo session created")

    def init_device_joints(self, dev_ee_names : dict):
        self.ee_names = dev_ee_names
        if self.joint_names is None:
            for dev, ee in dev_ee_names.items():
                dev.joint_ids, dev.joint_names = self.get_joints_in_ee_kinematic_tree([ee], dev.start_body_id)
                dev.init_joints()
        else:
            for dev, ee in dev_ee_names:
                dev.joint_ids = np.array([self.sim_model.name2id(name, 'joint') for name in self.joint_names])
                dev.init_joints()

    def init_viewer(self):
        if self.create_offscreen_rendercontext:
            render_surface = dm_render.Renderer(max_width=MAX_FRONTBUFFER_SIZE, max_height=MAX_FRONTBUFFER_SIZE)
            renderer = dm_view_renderer.OffScreenRenderer(self.sim_model, render_surface)
            renderer.components += dm_views.ViewportLayout()
        else:
            renderer = dm_view_renderer.NullRenderer()
        self.viewer.initialize(self.sim, renderer, touchpad=False)
        self.viewer.zoom_to_scene()

    def tick(self, update_display = True):
        """ Ref dm_control/dm_control/viewer/application.py - _tick() """
        self._viewport.set_size(*self.window.shape)
        #time_elapsed = self._frame_timer.tick() * self._time_multiplier.get()
        
        # move simulation ahead one time step
        self.sim.step()

        if self.viewer._renderer:
            self.time += self.dt
            self.timestep = int(self.time / self.dt)

            #freq_display = not self.timestep % self.display_frequency
            if self.visualize and update_display:
                self.viewer.render()
            
            return self.viewer._renderer.pixels
        else:
            return NULL_RENDERER.pixels

    def disconnect(self):
        """Stop and reset the simulation."""
        # nothing to do to close a MuJoCo session
        print("MuJoCo session closed...")

    def get_joints_in_ee_kinematic_tree(self, ee_names, base_body_id):
        """Get the names and ids of joints connecting the end-effector to the world"""
        parent_body_id = lambda body_id : self.model_ptr.body_parentid[body_id]

        # get the kinematic tree for the arm
        joint_ids = np.array([])
        joint_names = np.array([])
        for ee_name in ee_names:
            body_id = self.sim_model.name2id(ee_name, 'body')
            # start with the end-effector (EE) and work back to the world body
            while parent_body_id(body_id) != 0 and parent_body_id(body_id) != base_body_id:
                jntadrs_start = self.model_ptr.body_jntadr[body_id]
                #print(ee_name, 'joint', jntadrs_start, self.sim_model.id2name(jntadrs_start, 'joint'))

                tmp_ids = np.array([])
                tmp_names = np.array([])
                for ii in range(self.model_ptr.body_jntnum[body_id]):
                    tmp_ids = np.append(tmp_ids, jntadrs_start + ii).astype(int)
                    tmp_names = np.append(tmp_names, self.sim_model.id2name(tmp_ids[-1], 'joint'))
                joint_ids = np.append(joint_ids, tmp_ids[::-1]).astype(int)
                joint_names = np.append(joint_names, tmp_names[::-1])
                body_id = parent_body_id(body_id)
            
            # flip the list so it starts with the base of the arm / first joint
            joint_names = np.array(joint_names[::-1])
            joint_ids = np.array(joint_ids[::-1]).astype(int)

        return joint_ids, joint_names

    def get_xyz_by_id(self, id, object_type="body"):
        if object_type == "mocap":  # commonly queried to find target
            mocap_id = self.model_ptr.body_mocapid[id]
            xyz = self.data_ptr.mocap_pos[mocap_id]
        elif object_type == "body":
            xyz = self.data_ptr.xpos[id]
        elif object_type == "geom":
            xyz = self.data_ptr.geom_xpos[id]
        elif object_type == "site":
            xyz = self.data_ptr.site_xpos[id]
        else:
            raise Exception(f"get_xyz for {object_type} object type not supported")
        return np.copy(xyz)

    def get_xyz(self, name, object_type="body"):
        """Returns the xyz position of the specified object

        name: string
            name of the object you want the xyz position of
        object_type: string
            type of object you want the xyz position of
            Can be: mocap, body, geom, site
        """
        id = self.sim_model.name2id(name, object_type)
        return self.get_xyz_by_id(id, object_type)

    def get_mocap_xyz_by_id(self, body_id):
        mocap_id = self.model_ptr.body_mocapid[body_id]
        #print('mocapid', self.model_ptr.body_mocapid, 'body_mocapid', self.sim_model.body_mocapid)
        #assert(self.model_ptr.body_mocapid.all() == self.sim_model.body_mocapid.all())
        return self.data_ptr.mocap_pos[mocap_id]

    def get_mocap_xyz(self, name):
        """Return the position of a mocap object in the Mujoco environment.

        name: string
            the name of the mocap object
        """
        return self.get_mocap_xyz_by_id(self.sim_model.name2id(name, 'body'))

    def set_mocap_xyz_by_id(self, body_id, xyz):
        mocap_id = self.model_ptr.body_mocapid[body_id]
        #assert(self.model_ptr.body_mocapid.all() == self.sim_model.body_mocapid.all())
        self.data_ptr.mocap_pos[mocap_id] = xyz

    def set_mocap_xyz(self, name, xyz):
        """Set the position of a mocap object in the Mujoco environment.

        name: string
            the name of the object
        xyz: np.array
            the [x,y,z] location of the target [meters]
        """
        self.set_mocap_xyz_by_id(self.sim_model.name2id(name, 'body'), xyz)

    def get_orientation_by_id(self, id, as_quat=True, object_type="body"):
        quat = None
        xmat = None
        if object_type == "mocap":  # commonly queried to find target
            quat = self.get_mocap_orientation_by_id(id)
        elif object_type == "body":
            quat = self.data_ptr.xquat[id] # Shape (4,)
        elif object_type == "geom":
            xmat = self.data_ptr.geom_xmat[id] # Shape (9,)
            quat = transformations.quaternion_from_matrix(xmat.reshape((3, 3)))
        elif object_type == "site":
            xmat = self.data_ptr.site_xmat[id] # Shape (9,)
            quat = transformations.quaternion_from_matrix(xmat.reshape((3, 3)))
        else:
            raise Exception(
                f"get_orientation for {object_type} object type not supported"
            )
        return np.copy(quat) if as_quat else xmat.reshape((3, 3))

    def get_orientation(self, name, as_quat=True, object_type="body"):
        """Returns the orientation of an object as the [w x y z] quaternion [radians]

        Parameters
        ----------
        name: string
            the name of the object of interest
        object_type: string, Optional (Default: body)
            The type of mujoco object to get the orientation of.
            Can be: mocap, body, geom, site
        """
        id = self.sim_model.name2id(name, object_type)
        return self.get_orientation_by_id(id, as_quat, object_type)

    def get_rotation_matrix_by_id(self, id, object_type="body"):
        if object_type == "mocap":  # commonly queried to find target
            quat = self.get_mocap_orientation_by_id(id)
            mjp.mju_quat2Mat(xmat, quat)
        elif object_type == "body":
            xmat = self.data_ptr.xmat[id]
        elif object_type == "geom":
            xmat = self.data_ptr.geom_xmat[id]
        elif object_type == "site":
            xmat = self.data_ptr.site_xmat[id]
        else:
            raise Exception(
                f"get_rotation_matrix_by_id for {object_type} object type not supported"
            )
        return np.copy(xmat)

    def get_rotation_matrix(self, name, object_type="body"):
        """Returns the rotation matrix of an object as the [w x y z] quaternion [radians]

        Parameters
        ----------
        name: string
            the name of the object of interest
        object_type: string, Optional (Default: body)
            The type of mujoco object to get the orientation of.
            Can be: mocap, body, geom, site
        """
        id = self.sim_model.name2id(name, object_type)
        return self.get_rotation_matrix_by_id(id, object_type)
        
    def get_xvelp(self, device, name, object_type="body"):
        return device.xvelp(name, object_type)
    
    def get_sensor_data(self):
        """Returns sensor data
        """
        return self.data_ptr.sensordata
    
    def get_mocap_orientation_by_id(self, body_id):
        mocap_id = self.model_ptr.body_mocapid[body_id]
        return self.data_ptr.mocap_quat[mocap_id]

    def get_mocap_orientation(self, name):
        return self.get_mocap_orientation_by_id(self.sim_model.name2id(name, 'body'))

    def set_mocap_orientation_by_id(self, body_id, quat):
        mocap_id = self.model_ptr.body_mocapid[body_id]
        self.data_ptr.mocap_quat[mocap_id] = quat

    def set_mocap_orientation(self, name, quat):
        """Sets the orientation of an object in the Mujoco environment

        Sets the orientation of an object using the provided Euler angles.
        Angles must be in a relative xyz frame.

        Parameters
        ----------
        name: string
            the name of the object of interest
        quat: np.array
            the [w x y z] quaternion [radians] for the object.
        """
        self.set_mocap_orientation_by_id(self.sim_model.name2id(name, 'body'), quat)
    
    def send_forces(self, device_model, u, use_joint_dyn_addrs=True):
        """Apply the specified torque to the robot joints

        Apply the specified torque to the robot joints, move the simulation
        one time step forward, and update the position of the hand object.

        Parameters
        ----------
        u: np.array
            the torques to apply to the robot [Nm]
        update_display: boolean, Optional (Default:True)
            toggle for updating display
        use_joint_dyn_addrs: boolean
            set false to update the control signal for all actuators
        """

        # NOTE: the qpos_addr's are unrelated to the order of the motors
        # NOTE: assuming that the robot arm motors are the first len(u) values
        if use_joint_dyn_addrs:
            self.data_ptr.ctrl[device_model.joint_dyn_addrs] = u[:]
        else:
            self.data_ptr.ctrl[:] = u[:]

        # Update position of hand object
        ee_name = device_model.EE
        id = self.sim_model.name2id(ee_name, 'body')
        #feedback = self.get_feedback()
        #hand_xyz = self.robot_model.Tx(name=ee_name, q=feedback["q"])
        hand_xyz = self.data_ptr.xpos[id] #self.get_xyz(ee_name, 'body')
        self.set_mocap_xyz("hand", hand_xyz)

        # Update orientation of hand object
        #hand_quat = self.robot_model.quaternion(name=ee_name, q=feedback["q"])
        hand_quat = self.data_ptr.xquat[id] #self.get_orientation(ee_name, 'body')
        self.set_mocap_orientation("hand", hand_quat)

    def set_external_force(self, name, u_ext):
        """
        Applies an external force to a specified body

        Parameters
        ----------
        u_ext: np.array([x, y, z, alpha, beta, gamma])
            external force to apply [Nm]
        name: string
            name of the body to apply the force to
        """
        self.data_ptr.xfrc_applied[self.sim.model.name2id(name, 'body')] = u_ext

    def send_target_angles(self, device_model, q):
        """Move the robot to the specified configuration.

        Parameters
        ----------
        q: np.array
            configuration to move to [radians]
        """
        self.data_ptr.qpos[device_model.joint_pos_addrs] = np.copy(q)
        self.sim.forward()

    def set_joint_state(self, device_model, q, dq):
        """Move the robot to the specified configuration.

        Parameters
        ----------
        q: np.array
            configuration to move to [rad]
        dq: np.array
            joint velocities [rad/s]
        """

        self.data_ptr.qpos[device_model.joint_pos_addrs] = np.copy(q)
        self.data_ptr.qvel[device_model.joint_vel_addrs] = np.copy(dq)
        self.sim.forward()

    def get_feedback(self, device_model):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """

        self.q = np.copy(self.data_ptr.qpos[device_model.joint_pos_addrs])
        self.dq = np.copy(self.data_ptr.qvel[device_model.joint_vel_addrs])
        self.dqq = np.copy(self.data_ptr.qacc[device_model.joint_vel_addrs])

        return {"q": self.q, "dq": self.dq, "dqq": self.dqq}
