import os
from xml.etree import ElementTree

import mujoco as mjp
import dm_control.mujoco as dm_mujoco
import numpy as np

from abr_control.utils import download_meshes
from abr_control.interfaces.interface import Interface

# Ref: https://github.com/abr/abr_control/blob/main/abr_control/arms/mujoco_config.py
class MujocoModel:
    """A wrapper on the Mujoco simulator to generate all the kinematics and
    dynamics calculations necessary for controllers.
    """

    JNT_POS_LENGTH = {
        mjp.mjtJoint.mjJNT_FREE: 7,
        mjp.mjtJoint.mjJNT_BALL: 4,
        mjp.mjtJoint.mjJNT_SLIDE: 1,
        mjp.mjtJoint.mjJNT_HINGE: 1,
    }

    JNT_DYN_LENGTH = {
        mjp.mjtJoint.mjJNT_FREE: 6,
        mjp.mjtJoint.mjJNT_BALL: 3,
        mjp.mjtJoint.mjJNT_SLIDE: 1,
        mjp.mjtJoint.mjJNT_HINGE: 1,
    }

    def __init__(self, sim_interface, xml_file=None, folder=None, use_sim_state=True, force_download=False):
        """Loads the Mujoco model from the specified xml file

        Parameters
        ----------
        xml_file: string
            the name of the arm model to load. If folder remains as None,
            the string passed in is parsed such that everything up to the first
            underscore is used for the arm directory, and the full string is
            used to load the xml within that folder.

            EX: 'myArm' and 'myArm_with_gripper' will both look in the
            'myArm' directory, however they will load myArm.xml and
            myArm_with_gripper.xml, respectively

            If a folder is passed in, then folder/xml_file is used
        folder: string, Optional (Default: None)
            specifies what folder to find the xml_file, if None specified will
            checking in abr_control/arms/xml_file (see above for xml_file)
        use_sim_state: Boolean, optional (Default: True)
            If set False, the state information is provided by the user, which
            is then used to calculate the corresponding dynamics values.
            The state is then set back to the sim state prior to the user
            provided state.
            If set true, any q and dq values passed in to the functions are
            ignored, and the current state of the simulator is used to
            calculate all functions. This can speed up the simulation, because
            the step of resetting the state on every call is omitted.
        force_download: boolean, Optional (Default: False)
            True to force downloading the mesh and texture files, useful when new files
            are added that may be missing.
            False: if the meshes folder is missing it will ask the user whether they
            want to download them
        """
        self.name = None
        self.use_sim_state = use_sim_state
        self.N_GRIPPEPR_JOINTS = 0

        self.sim_interface = sim_interface
        self.sim = None
        self.sim_model = sim_interface.sim_model
        self.model_ptr = None
        self.data_ptr = None

        self.joint_ids = np.array([])
        self.joint_ids_all = np.array([])
        self.joint_names = np.array([])

        if os.path.isabs(xml_file):
            self.xml_file = xml_file
            self.xml_dir = os.path.dirname(xml_file) if (folder is None and xml_file is not None) else folder
        elif folder:
            self.xml_dir = folder
            self.xml_file = os.path.join(self.xml_dir, xml_file)
        else:
            arm_dir = xml_file.split("_")[0]
            current_dir = os.path.dirname(__file__)
            self.xml_file = os.path.join(current_dir, arm_dir, f"{xml_file}.xml")
            self.xml_dir = os.path.join(current_dir, arm_dir)

        # Init configs
        self.init_configs(force_download)

        # Init default sim interface as loaded from [self.xml_file]
        if self.sim_model is None:
            self.set_sim(dm_mujoco.Physics.from_xml_path(self.xml_file))
            
        
    def init_configs(self, force_download):
        if self.xml_file is None:
            return

        # get access to some of our custom arm parameters from the xml definition
        tree = ElementTree.parse(self.xml_file)
        root = tree.getroot()
        for custom in root.findall("custom/numeric"):
            name = custom.get("name")
            if name == "START_ANGLES":
                START_ANGLES = custom.get("data").split(" ")
                self.START_ANGLES = np.array([float(angle) for angle in START_ANGLES])
            elif name == "N_GRIPPER_JOINTS":
                self.N_GRIPPER_JOINTS = int(custom.get("data"))

        # check for google_id specifying download location of robot mesh files
        self.google_id = None
        for custom in root.findall("custom/text"):
            name = custom.get("name")
            if name == "google_id":
                self.google_id = custom.get("data")

        # check if the user has downloaded the required mesh files
        # if not prompt them to do so
        if self.google_id is not None:
            # get list of expected files to check if all have been downloaded
            files = []
            for asset in root.findall("asset/mesh"):
                files.append(asset.get("file"))

            for asset in root.findall("asset/texture"):
                # assuming that texture are placed in the meshes folder
                files.append(asset.get("file").split("/")[1])

            # check if our mesh folder exists, then check we have all the files
            download_meshes.check_and_download(
                name=self.xml_dir + "/meshes",
                google_id=self.google_id,
                force_download=force_download,
                files=files,
            )

    def set_sim(self, sim):
        """Called only once the Mujoco simulation is created,
        this connects the config to the simulator so it can access the
        kinematics and dynamics information calculated by Mujoco.

        Parameters
        ----------
        sim: MjSim
            The Mujoco Simulator object created by the Mujoco Interface class
        """
        self.sim = sim
        self.sim_model = self.sim.model # dm_control model
        self.model_ptr = self.sim_model.ptr # mujoco model
        self.data_ptr = self.sim.data.ptr # mujoco model data
        self.N_JOINTS = self.model_ptr.njnt

    def init_joints(self):
        if self.joint_ids is []:
            return
        self.joint_ids_all = self.joint_ids

        print(f'{self.name}: joint_ids:{self.joint_ids}')
        self.joint_types = [self.model_ptr.jnt_type[id] for id in self.joint_ids]
        print(f'{self.name}: joint_types:{self.joint_types}')
        self.joint_pos_addrs = [self.model_ptr.jnt_qposadr[id] for id in self.joint_ids]
        self.joint_vel_addrs = [self.model_ptr.jnt_dofadr[id] for id in self.joint_ids]
        
        joint_pos_addrs = []
        for elem in self.joint_pos_addrs:
            if isinstance(elem, tuple):
                joint_pos_addrs += list(range(elem[0], elem[1]))
            else:
                joint_pos_addrs.append(elem)
        self.joint_pos_addrs = joint_pos_addrs
        print(f'{self.name}: joint_pos_addrs:{joint_pos_addrs}')

        joint_vel_addrs = []
        for elem in self.joint_vel_addrs:
            if isinstance(elem, tuple):
                joint_vel_addrs += list(range(elem[0], elem[1]))
            else:
                joint_vel_addrs.append(elem)
        self.joint_vel_addrs = joint_vel_addrs
        print(f'{self.name}: joint_vel_addrs:{joint_vel_addrs}')

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
            if ii in self.joint_ids:
                self.joint_dyn_addrs.append(index)
                if joint_type == mjp.mjtJoint.mjJNT_FREE:  # free joint
                    self.joint_dyn_addrs += [jj + index for jj in range(1, 6)]
                    index += 6  # derivative has 6 dimensions
                elif joint_type == mjp.mjtJoint.mjJNT_BALL:  # ball joint
                    self.joint_dyn_addrs += [jj + index for jj in range(1, 3)]
                    index += 3  # derivative has 3 dimension
                else:  # slide or hinge joint
                    index += 1  # derivative has 1 dimensions

        # give the robot config access to the sim for wrapping the
        # forward kinematics / dynamics functions
        print(f'{self.name}: joint_dyn_addrs:{self.joint_dyn_addrs}')
        print(f'{self.name}: All joints info fetched')

    def calc_joints_data(self):
        """Calculate joints data
        """
        # get access to the Mujoco simulation
        # number of controllable joints in the robot arm
        self.N_JOINTS = len(self.joint_pos_addrs)
        # number of joints in the Mujoco simulation
        self.N_ALL_JOINTS = self.model_ptr.nv #njnt
        N_ALL_JOINTS = self.N_ALL_JOINTS
        print('N_JOINTS', self.N_JOINTS, 'N_ALL_JOINTS', N_ALL_JOINTS)

        # need to calculate the joint_vel_addrs indices in flat vectors returned
        # for the Jacobian
        print(self.joint_vel_addrs)
        
        self.jac_indices = np.hstack(
            # 6 because position and rotation Jacobians are 3 x N_JOINTS
            [self.joint_vel_addrs + [(i * N_ALL_JOINTS) for i in range(3)]]
        )
        print('jac_indices', self.jac_indices)

        # for the inertia matrix
        self.M_indices = [
            i * N_ALL_JOINTS + j
            for j in self.joint_vel_addrs
            for i in self.joint_vel_addrs
        ]

        # a place to store data returned from Mujoco
        self._g = np.zeros(self.N_JOINTS)
        self._J3NP = np.zeros((3, N_ALL_JOINTS))
        self._J3NR = np.zeros((3, N_ALL_JOINTS))
        self._J6N = np.zeros((6, self.N_JOINTS))
        self._MNN = np.zeros((N_ALL_JOINTS, N_ALL_JOINTS))
        self._R9 = np.zeros(9)
        self._R = np.zeros((3, 3))
        self._x = np.ones(4)

    def _load_state(self, q, dq=None, u=None):
        """Change the current joint angles

        Parameters
        ----------
        q: np.array
            The set of joint angles to move the arm to [rad]
        dq: np.array
            The set of joint velocities to move the arm to [rad/sec]
        u: np.array
            The set of joint forces to apply to the arm joints [Nm]
        """
        # save current state
        old_q = np.copy(self.data_ptr.qpos[self.joint_pos_addrs])
        old_dq = np.copy(self.data_ptr.qvel[self.joint_vel_addrs])
        old_u = np.copy(self.data_ptr.ctrl)

        # update positions to specified state
        self.data_ptr.qpos[self.joint_pos_addrs] = np.copy(q)
        if dq is not None:
            self.data_ptr.qvel[self.joint_vel_addrs] = np.copy(dq)
        if u is not None:
            self.data_ptr.ctrl[:] = np.copy(u)

        # move simulation forward to calculate new kinematic information
        self.sim.forward()

        return old_q, old_dq, old_u

    def g(self, q=None):
        """Returns qfrc_bias variable, which stores the effects of Coriolis,
        centrifugal, and gravitational forces

        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        # TODO: For the Coriolis and centrifugal functions, setting the
        # velocity before calculation is important, how best to do this?
        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        g = -1 * self.data_ptr.qfrc_bias[self.joint_vel_addrs]

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

        return g

    def dJ(self, name, q=None, dq=None, x=None):
        """Returns the derivative of the Jacobian wrt to time

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        dq: float numpy.array, optional (Default: None)
            The joint velocities of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        # Note from Emo in Mujoco forums:
        # 'You would have to use a finate-difference approximation in the
        # general case, check differences.cpp'
        raise NotImplementedError

    def J(self, name, q=None, x=None, object_type="body"):
        """Returns the Jacobian for the specified Mujoco object

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        object_type: string, the Mujoco object type, optional (Default: body)
            options: body, geom, site
        """
        if x is not None and not np.allclose(x, 0):
            raise Exception("x offset currently not supported, set to None")

        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        id = self.sim_model.name2id(name, object_type)
        if object_type == "body":
            # TODO: test if using this function is faster than the old way
            # NOTE: for bodies, the Jacobian for the COM is returned
            mjp.mj_jacBodyCom(self.model_ptr, self.data_ptr, self._J3NP, self._J3NR, id)
        else:
            if object_type == "geom":
                jacp = mjp.mj_jacGeom(self.model_ptr, self.data_ptr, self._J3NP, 0, id) # self.data.get_geom_jacp
                jacr = mjp.mj_jacGeom(self.model_ptr, self.data_ptr, 0, self._J3NR, id) # self.data.get_geom_jacr
            elif object_type == "site":
                jacp = mjp.mj_jacSite(self.model_ptr, self.data_ptr, self._J3NP, 0, id) # self.data.get_site_jacp
                jacr = mjp.mj_jacSite(self.model_ptr, self.data_ptr, 0, self._J3NR, id) # self.data.get_site_jacr
            else:
                raise Exception("Invalid object type specified: ", object_type)

            jacp(name, self._J3NP)[self.jac_indices]  # pylint: disable=W0106
            jacr(name, self._J3NR)[self.jac_indices]  # pylint: disable=W0106

        # get the position Jacobian hstacked (1 x N_JOINTS*3)
        self._J6N[:3] = self._J3NP[:, self.joint_vel_addrs].reshape((3, self.N_JOINTS))
        # get the rotation Jacobian hstacked (1 x N_JOINTS*3)
        self._J6N[3:] = self._J3NR[:, self.joint_vel_addrs].reshape((3, self.N_JOINTS))

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

        return np.copy(self._J6N)

    def M(self, q=None):
        """Returns the inertia matrix in task space

        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        # stored in mjData.qM, stored in custom sparse format,
        # convert qM to a dense matrix with mj_fullM
        mjp.mj_fullM(self.model_ptr, self._MNN, self.data_ptr.qM)
        M = self._MNN[self.joint_vel_addrs][:, self.joint_vel_addrs]

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

        return np.copy(M)

    def R(self, name, q=None, object_type="body"):
        """Returns the rotation matrix of the specified body

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        id = self.sim_model.name2id(name, 'body')
        if object_type == "body":
            #mjp.mju_quat2Mat(self._R9, self.data_ptr.xquat[id])
            self._R9 = self.data_ptr.xmat[id]
            # OR
            # mujoco.mju_quat2Mat(self._R9, self.data.body(name).xquat)
        elif object_type == "geom":
            self._R9 = self.data_ptr.geom_xmat[id] # self.data.geom(name).xmat
        elif object_type == "site":
            self._R9 = self.data_ptr.site_xmat[id] # self.data.site(name).xmat
        else:
            raise Exception("Invalid object type specified: ", object_type)

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

        self._R = self._R9.reshape((3, 3))
        return self._R

    def quaternion(self, name, q=None):
        """Returns the quaternion of the specified body
        Parameters
        ----------

        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        quaternion = np.copy(self.data_ptr.xquat[self.sim.model.name2id(name, 'body')])
        # Or np.copy(self.data.body(name).xquat) if self.data is mujoco.MjData(self.model)

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

        return quaternion

    def C(self, q=None, dq=None):
        """NOTE: The Coriolis and centrifugal effects (and gravity) are
        already accounted for by Mujoco in the qfrc_bias variable. There's
        no easy way to separate these, so all are returned by the g function.
        To prevent accounting for these effects twice, this function will
        return an error instead of qfrc_bias again.
        """
        raise NotImplementedError(
            "Coriolis and centrifugal effects already accounted "
            + "for in the term return by the gravity function."
        )

    def T(self, name, q=None, x=None):
        """Get the transform matrix of the specified body.

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError

    def Tx(self, name, q=None, x=None, object_type="body"):
        """Returns the Cartesian coordinates of the specified Mujoco body

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        object_type: string, the Mujoco object type, optional (Default: body)
            options: body, geom, site, camera, light, mocap
        """
        if x is not None and not np.allclose(x, 0):
            raise Exception("x offset currently not supported: ", x)

        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        id = self.sim_model.name2id(name, object_type)
        if object_type == "body":
            Tx = np.copy(self.data_ptr.xpos[id]) # np.copy(self.data.body(name).xpos)
        elif object_type == "geom":
            Tx = np.copy(self.data_ptr.geom_xpos[id]) # np.copy(self.data.geom(name).xpos)
        elif object_type == "joint":
            Tx = np.copy(self.data_ptr.xanchor[id]) # np.copy(self.data.joint(name).xanchor)
        elif object_type == "site":
            Tx = np.copy(self.data_ptr.site_xpos[id]) # np.copy(self.data.site(name).xpos)
        elif object_type == "camera":
            Tx = np.copy(self.data_ptr.cam_xpos[id]) # np.copy(self.data.com(name).xpos)
        elif object_type == "light":
            Tx = np.copy(self.data_ptr.light_xpos[id]) # np.copy(self.data.light(name).xpos)
        elif object_type == "mocap":
            mocap_id = self.model_ptr.body_mocapid[id] # np.copy(self.data.mocap(name).pos)
            Tx = np.copy(self.data_ptr.mocap_pos[mocap_id])
        else:
            raise Exception("Invalid object type specified: ", object_type)

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

        return Tx

    def T_inv(self, name, q=None, x=None):
        """Get the inverse transform matrix of the specified body.

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError

    def xvelp(self, name, object_type="body"):
        id = self.sim_model.name2id(name, object_type)
        if object_type == "mocap":  # commonly queried to find target
            pass
        elif object_type == "body":
            self.J(name)
            xvelp = np.dot(self._J3NP, self.data_ptr.qvel)
            return xvelp
        elif object_type == "geom":
            pass
        elif object_type == "site":
            pass
        else:
            raise Exception(
                f"get_xvelp for {object_type} object type not supported"
            )
        return None