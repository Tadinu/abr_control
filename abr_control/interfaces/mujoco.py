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

from abr_control.utils import transformations

from .interface import Interface

MAX_FRONTBUFFER_SIZE = 2048
NULL_RENDERER = dm_view_renderer.NullRenderer()
class AbrMujoco(Interface):
    """An interface for MuJoCo using the mujoco package.

    Parameters
    ----------
    robot_config: class instance
        contains all relevant information about the arm
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
        robot_config,
        dt=0.001,
        visualize=True,
        create_offscreen_rendercontext=False,
    ):

        super().__init__(robot_config)

        self.dt = dt  # time step
        self.count = 0  # keep track of how many times send forces is called

        self.robot_config = robot_config
        # set the time step for simulation
        self.robot_config.model.opt.timestep = self.dt

        # turns the visualization on or off
        self.visualize = visualize
        # if we want the offscreen render context
        self.create_offscreen_rendercontext = create_offscreen_rendercontext

    def connect(self, joint_names=None, camera_id=-1, **kwargs):
        """
        joint_names: list, optional (Default: None)
            list of joint names to send control signal to and get feedback from
            if None, the joints in the kinematic tree connecting the end-effector
            to the world are used
        camera_id: int, optional (Default: -1)
            the id of the camera to use for the visualization
        """
        self.sim = dm_mujoco.Physics.from_xml_path(self.robot_config.xml_file)
        self.sim.forward()  # run forward to fill in sim.data
        model = self.sim.model
        self.model = model
        self.model_ptr = self.model.ptr
        self.data_ptr = self.sim.data.ptr
        self.robot_config.update_model_from_interface(self)

        if joint_names is None:
            joint_ids, joint_names = self.get_joints_in_ee_kinematic_tree()
        else:
            joint_ids = [self.model.name2id(name) for name in joint_names]
        print(f'joint_ids:{joint_ids}')
        print(f'joint_names:{joint_names}')
        self.joint_types = [self.model_ptr.jnt_type[id] for id in joint_ids]
        print(f'joint_types:{self.joint_types}')
        self.joint_pos_addrs = [self.model_ptr.jnt_qposadr[id] for id in joint_ids]
        self.joint_vel_addrs = [self.model_ptr.jnt_dofadr[id] for id in joint_ids]

        joint_pos_addrs = []
        for elem in self.joint_pos_addrs:
            if isinstance(elem, tuple):
                joint_pos_addrs += list(range(elem[0], elem[1]))
            else:
                joint_pos_addrs.append(elem)
        self.joint_pos_addrs = joint_pos_addrs
        print(f'joint_pos_addrs:{joint_pos_addrs}')

        joint_vel_addrs = []
        for elem in self.joint_vel_addrs:
            if isinstance(elem, tuple):
                joint_vel_addrs += list(range(elem[0], elem[1]))
            else:
                joint_vel_addrs.append(elem)
        self.joint_vel_addrs = joint_vel_addrs
        print(f'joint_vel_addrs:{joint_vel_addrs}')

        # Need to also get the joint rows of the Jacobian, inertia matrix, and
        # gravity vector. This is trickier because if there's a quaternion in
        # the joint (e.g. a free joint or a ball joint) then the joint position
        # address will be different than the joint Jacobian row. This is because
        # the quaternion joint will have a 4D position and a 3D derivative. So
        # we go through all the joints, and find out what type they are, then
        # calculate the Jacobian position based on their order and type.
        index = 0
        self.joint_dyn_addrs = []
        for ii, joint_type in enumerate(self.joint_types):
            if ii in joint_ids:
                self.joint_dyn_addrs.append(index)
                if joint_type == 0:  # free joint
                    self.joint_dyn_addrs += [jj + index for jj in range(1, 6)]
                    index += 6  # derivative has 6 dimensions
                elif joint_type == 1:  # ball joint
                    self.joint_dyn_addrs += [jj + index for jj in range(1, 3)]
                    index += 3  # derivative has 3 dimension
                else:  # slide or hinge joint
                    index += 1  # derivative has 1 dimensions

        # give the robot config access to the sim for wrapping the
        # forward kinematics / dynamics functions
        print(f'joint_dyn_addrs:{self.joint_dyn_addrs}')
        self.robot_config._connect(
            self.joint_pos_addrs, self.joint_vel_addrs, self.joint_dyn_addrs
        )

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

    def init_viewer(self):
        if self.create_offscreen_rendercontext:
            render_surface = dm_render.Renderer(max_width=MAX_FRONTBUFFER_SIZE, max_height=MAX_FRONTBUFFER_SIZE)
            renderer = dm_view_renderer.OffScreenRenderer(self.model, render_surface)
            renderer.components += dm_views.ViewportLayout()
        else:
            renderer = dm_view_renderer.NullRenderer()
        self.viewer.initialize(self.sim, renderer, touchpad=False)
        self.viewer.zoom_to_scene()

    def tick(self):
        """ Ref dm_control/dm_control/viewer/application.py - _tick() """
        self._viewport.set_size(*self.window.shape)
        #time_elapsed = self._frame_timer.tick() * self._time_multiplier.get()
        if self.viewer._renderer:
            self.viewer.render()
            return self.viewer._renderer.pixels
        else:
            return NULL_RENDERER.pixels

    def disconnect(self):
        """Stop and reset the simulation."""
        # nothing to do to close a MuJoCo session
        print("MuJoCo session closed...")

    def get_joints_in_ee_kinematic_tree(self):
        """Get the names and ids of joints connecting the end-effector to the world"""
        # get the kinematic tree for the arm
        joint_ids = []
        joint_names = []
        body_id = self.model.name2id('EE', 'body')
        # start with the end-effector (EE) and work back to the world body
        while self.model_ptr.body_parentid[body_id] != 0:
            jntadrs_start = self.model_ptr.body_jntadr[body_id]
            tmp_ids = []
            tmp_names = []
            for ii in range(self.model_ptr.body_jntnum[body_id]):
                tmp_ids.append(jntadrs_start + ii)
                tmp_names.append(self.model.id2name(tmp_ids[-1], 'joint'))
            joint_ids += tmp_ids[::-1]
            joint_names += tmp_names[::-1]
            body_id = self.model_ptr.body_parentid[body_id]
        # flip the list so it starts with the base of the arm / first joint
        joint_names = joint_names[::-1]
        joint_ids = np.array(joint_ids[::-1])

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
        id = self.model.name2id(name, object_type)
        return self.get_xyz_by_id(id, object_type)

    def get_mocap_xyz_by_id(self, body_id):
        mocap_id = self.model_ptr.body_mocapid[body_id]
        #print('mocapid', self.model_ptr.body_mocapid, 'body_mocapid', self.model.body_mocapid)
        #assert(self.model_ptr.body_mocapid.all() == self.model.body_mocapid.all())
        return self.data_ptr.mocap_pos[mocap_id]

    def get_mocap_xyz(self, name):
        """Return the position of a mocap object in the Mujoco environment.

        name: string
            the name of the mocap object
        """
        return self.get_mocap_xyz_by_id(self.model.name2id(name, 'body'))

    def set_mocap_xyz_by_id(self, body_id, xyz):
        mocap_id = self.model_ptr.body_mocapid[body_id]
        #assert(self.model_ptr.body_mocapid.all() == self.model.body_mocapid.all())
        self.data_ptr.mocap_pos[mocap_id] = xyz

    def set_mocap_xyz(self, name, xyz):
        """Set the position of a mocap object in the Mujoco environment.

        name: string
            the name of the object
        xyz: np.array
            the [x,y,z] location of the target [meters]
        """
        self.set_mocap_xyz_by_id(self.model.name2id(name, 'body'), xyz)

    def get_orientation_by_id(self, id, object_type="body"):
        if object_type == "mocap":  # commonly queried to find target
            quat = self.get_mocap_orientation_by_id(id)
        elif object_type == "body":
            quat = self.data_ptr.xquat[id]
        elif object_type == "geom":
            xmat = self.data_ptr.geom_xmat[id]
            quat = transformations.quaternion_from_matrix(xmat.reshape((3, 3)))
        elif object_type == "site":
            xmat = self.data_ptr.site_xmat[id]
            quat = transformations.quaternion_from_matrix(xmat.reshape((3, 3)))
        else:
            raise Exception(
                f"get_orientation for {object_type} object type not supported"
            )
        return np.copy(quat)

    def get_orientation(self, name, object_type="body"):
        """Returns the orientation of an object as the [w x y z] quaternion [radians]

        Parameters
        ----------
        name: string
            the name of the object of interest
        object_type: string, Optional (Default: body)
            The type of mujoco object to get the orientation of.
            Can be: mocap, body, geom, site
        """
        id = self.model.name2id(name, object_type)
        return self.get_orientation_by_id(id, object_type)

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
        id = self.model.name2id(name, object_type)
        return self.get_rotation_matrix_by_id(id, object_type)
        
    def get_mocap_orientation_by_id(self, body_id):
        mocap_id = self.model_ptr.body_mocapid[body_id]
        return self.data_ptr.mocap_quat[mocap_id]

    def get_mocap_orientation(self, name):
        return self.get_mocap_orientation_by_id(self.model.name2id(name, 'body'))

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
        self.set_mocap_orientation_by_id(self.model.name2id(name, 'body'), quat)

    def send_forces(self, u, update_display=True):
        """Apply the specified torque to the robot joints

        Apply the specified torque to the robot joints, move the simulation
        one time step forward, and update the position of the hand object.

        Parameters
        ----------
        u: np.array
            the torques to apply to the robot [Nm]
        update_display: boolean, Optional (Default:True)
            toggle for updating display
        """

        # NOTE: the qpos_addr's are unrelated to the order of the motors
        # NOTE: assuming that the robot arm motors are the first len(u) values
        self.data_ptr.ctrl[:] = u[:]

        # move simulation ahead one time step
        self.sim.step()

        # Update position of hand object
        id = self.model.name2id('EE', 'body')
        #feedback = self.get_feedback()
        #hand_xyz = self.robot_config.Tx(name="EE", q=feedback["q"])
        hand_xyz = self.data_ptr.xpos[id] #self.get_xyz('EE', 'body')
        self.set_mocap_xyz("hand", hand_xyz)

        # Update orientation of hand object
        #hand_quat = self.robot_config.quaternion(name="EE", q=feedback["q"])
        hand_quat = self.data_ptr.xquat[id] #self.get_orientation('EE', 'body')
        self.set_mocap_orientation("hand", hand_quat)

        if self.visualize and update_display:
            self.viewer.render()
        self.count += self.dt

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

    def send_target_angles(self, q):
        """Move the robot to the specified configuration.

        Parameters
        ----------
        q: np.array
            configuration to move to [radians]
        """

        self.data_ptr.qpos[self.joint_pos_addrs] = np.copy(q)
        self.sim.forward()

    def set_joint_state(self, q, dq):
        """Move the robot to the specified configuration.

        Parameters
        ----------
        q: np.array
            configuration to move to [rad]
        dq: np.array
            joint velocities [rad/s]
        """

        self.data_ptr.qpos[self.joint_pos_addrs] = np.copy(q)
        self.data_ptr.qvel[self.joint_vel_addrs] = np.copy(dq)
        self.sim.forward()

    def get_feedback(self):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """

        self.q = np.copy(self.data_ptr.qpos[self.joint_pos_addrs])
        self.dq = np.copy(self.data_ptr.qvel[self.joint_vel_addrs])

        return {"q": self.q, "dq": self.dq}
