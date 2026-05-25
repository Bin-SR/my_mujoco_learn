#!/usr/bin/env python3
"""
MuJoCo environment wrapper for Franka Emika Panda robot.

Loads the Panda model from mujoco_menagerie and provides a clean
interface for simulation stepping, state reading, and rendering.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import mujoco
from mujoco import viewer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model asset resolution
# ---------------------------------------------------------------------------

def _find_menagerie_path() -> Optional[Path]:
    """Locate the mujoco_menagerie assets directory."""
    # 1. MMC_ASSETS environment variable (common convention)
    env_path = os.environ.get('MMC_ASSETS', '')
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # 2. pip-installed mujoco-menagerie package
    try:
        import mujoco_menagerie
        # site-packages/mujoco_menagerie/__init__.py -> package dir
        return Path(mujoco_menagerie.__file__).parent
    except ImportError:
        pass

    # 3. Check typical mujoco_menagerie clone locations
    candidates = [
        Path.home() / 'mujoco_menagerie',
        Path.home() / '.mujoco' / 'mujoco_menagerie',
        Path('/opt/mujoco_menagerie'),
    ]
    for p in candidates:
        if p.exists():
            return p

    return None


def _resolve_panda_xml(menagerie_path: Optional[Path] = None,
                       local_models: Optional[Path] = None) -> str:
    """
    Resolve the absolute path to the Panda scene XML.

    Priority:
      1. Explicit menagerie_path argument
      2. Local models/ directory (bundled copy)
      3. Auto-detected mujoco_menagerie install
    """
    if menagerie_path is None:
        menagerie_path = _find_menagerie_path()

    if menagerie_path is not None:
        scene_xml = menagerie_path / 'franka_emika_panda' / 'scene.xml'
        if scene_xml.exists():
            logger.info(f'Using menagerie scene: {scene_xml}')
            return str(scene_xml)

        panda_xml = menagerie_path / 'franka_emika_panda' / 'panda.xml'
        if panda_xml.exists():
            logger.info(f'Using menagerie panda: {panda_xml}')
            return str(panda_xml)

    if local_models is not None:
        local_scene = local_models / 'panda_scene.xml'
        if local_scene.exists():
            logger.info(f'Using local scene: {local_scene}')
            return str(local_scene)

    raise FileNotFoundError(
        'Could not find Franka Emika Panda model.\n'
        'Install mujoco_menagerie:  pip install mujoco-menagerie\n'
        'Or set MMC_ASSETS env to the menagerie root directory.'
    )


# ---------------------------------------------------------------------------
# Joint / actuator mapping for the Panda arm
# ---------------------------------------------------------------------------

# Standard Panda arm joint names (order matches menagerie model)
PANDA_ARM_JOINTS = [
    'panda_joint1',   # base rotation
    'panda_joint2',   # shoulder
    'panda_joint3',   # elbow
    'panda_joint4',   # forearm roll
    'panda_joint5',   # wrist flex
    'panda_joint6',   # wrist roll
    'panda_joint7',   # wrist yaw
]

PANDA_GRIPPER_JOINTS = [
    'panda_finger_joint1',
    'panda_finger_joint2',
]

PANDA_ALL_JOINTS = PANDA_ARM_JOINTS + PANDA_GRIPPER_JOINTS


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class MujocoPandaEnv:
    """
    A self-contained MuJoCo simulation environment for the Franka Emika Panda.

    Usage::

        env = MujocoPandaEnv()
        env.reset()
        for _ in range(1000):
            env.step()
            joints = env.get_joint_positions()
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 menagerie_path: Optional[str] = None,
                 local_models: Optional[str] = None,
                 render_mode: str = 'window',
                 timestep: float = 0.002,
                 headless: bool = False):
        """
        Parameters
        ----------
        model_path : str, optional
            Direct path to an MJCF / XML file. Overrides auto-detection.
        menagerie_path : str, optional
            Path to mujoco_menagerie root.
        local_models : str, optional
            Path to local models/ directory with bundled XML files.
        render_mode : str
            One of `'window'`, `'offscreen'`.
        timestep : float
            Simulation timestep in seconds.
        headless : bool
            If True, skip all rendering (useful for headless servers).
        """
        self._timestep = timestep
        self._render_mode = render_mode
        self._headless = headless

        # Resolve model XML
        if model_path is not None:
            xml_path = model_path
        else:
            mp = Path(menagerie_path) if menagerie_path else None
            lp = Path(local_models) if local_models else None
            xml_path = _resolve_panda_xml(menagerie_path=mp, local_models=lp)

        logger.info(f'Loading MuJoCo model from: {xml_path}')
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)

        # Apply timestep override
        self._model.opt.timestep = self._timestep

        # Build joint / actuator index maps
        self._build_index_maps()

        # Viewer (created lazily or on demand)
        self._viewer: Optional[viewer.Handle] = None
        self._renderer: Optional[mujoco.Renderer] = None

        # Simulation state
        self._step_count = 0
        self._start_time = time.time()

        logger.info(
            f'MuJoCo Panda env ready. '
            f'Joints: {self.nj}  Actuators: {self.nu}  '
            f'timestep={self._timestep}s'
        )

    # ---- index maps -------------------------------------------------------

    def _build_index_maps(self):
        """Map joint/actuator names to MuJoCo IDs."""
        self.joint_name_to_id: Dict[str, int] = {}
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_name_to_id[name] = i

        self.actuator_name_to_id: Dict[str, int] = {}
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_name_to_id[name] = i

        # Verify known joints are present
        found = [j for j in PANDA_ALL_JOINTS if j in self.joint_name_to_id]
        if found:
            logger.info(f'Found {len(found)}/{len(PANDA_ALL_JOINTS)} expected Panda joints')
        else:
            logger.warning(
                'No standard Panda joint names detected. '
                'Joint map built from model directly.'
            )

    # ---- properties -------------------------------------------------------

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def nj(self) -> int:
        return self._model.njnt

    @property
    def nu(self) -> int:
        return self._model.nu

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def sim_time(self) -> float:
        return self._data.time

    # ---- simulation loop -------------------------------------------------

    def step(self, ctrl: Optional[np.ndarray] = None) -> None:
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        ctrl : np.ndarray, optional
            Actuator control signals. Shape `(nu,)`.
            If None, zero-control is applied (robot holds position).
        """
        if ctrl is not None:
            self._data.ctrl[:] = ctrl
        else:
            self._data.ctrl[:] = 0.0

        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

    def step_n(self, n: int, ctrl: Optional[np.ndarray] = None) -> None:
        """Step `n` times with the same control."""
        for _ in range(n):
            self.step(ctrl)

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        mujoco.mj_resetData(self._model, self._data)
        self._step_count = 0
        self._start_time = time.time()

    # ---- state access -----------------------------------------------------

    def get_qpos(self) -> np.ndarray:
        """Return joint positions (qpos)."""
        return self._data.qpos.copy()

    def get_qvel(self) -> np.ndarray:
        """Return joint velocities (qvel)."""
        return self._data.qvel.copy()

    def get_joint_positions(self, joint_names: Optional[list] = None) -> Dict[str, float]:
        """
        Return a dict mapping joint name -> position (radians).

        Parameters
        ----------
        joint_names : list, optional
            Specific joint names. Defaults to all joints.
        """
        names = joint_names or list(self.joint_name_to_id.keys())
        result = {}
        for name in names:
            jid = self.joint_name_to_id.get(name)
            if jid is not None:
                qpos_addr = self._model.jnt_qposadr[jid]
                result[name] = float(self._data.qpos[qpos_addr])
        return result

    def get_joint_velocities(self, joint_names: Optional[list] = None) -> Dict[str, float]:
        """Return a dict mapping joint name -> velocity (rad/s)."""
        names = joint_names or list(self.joint_name_to_id.keys())
        result = {}
        for name in names:
            jid = self.joint_name_to_id.get(name)
            if jid is not None:
                dof_addr = self._model.jnt_dofadr[jid]
                result[name] = float(self._data.qvel[dof_addr])
        return result

    def get_body_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (position, quaternion) of a body in world frame.

        position : np.ndarray (3,)  – x, y, z
        quaternion : np.ndarray (4,) – w, x, y, z
        """
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f'Body ""{body_name}"" not found in model.')
        pos = self._data.xpos[bid].copy()
        quat = self._data.xquat[bid].copy()  # w, x, y, z
        return pos, quat

    def get_site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (position, rotation_matrix) of a site in world frame.
        """
        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise ValueError(f'Site ""{site_name}"" not found in model.')
        pos = self._data.site_xpos[sid].copy()
        rot = self._data.site_xmat[sid].copy().reshape(3, 3)
        return pos, rot

    # ---- control ----------------------------------------------------------

    def set_control(self, ctrl: np.ndarray) -> None:
        """Write control signals to the data buffer (applied on next step)."""
        self._data.ctrl[:] = ctrl

    def get_actuator_force(self) -> np.ndarray:
        """Return actuator forces."""
        return self._data.actuator_force.copy()

    # ---- rendering --------------------------------------------------------

    def launch_viewer(self) -> Optional[viewer.Handle]:
        """
        Launch the interactive MuJoCo passive viewer.

        Returns the viewer handle, or None if headless / already open.
        """
        if self._headless:
            return None
        if self._viewer is not None:
            return self._viewer
        try:
            self._viewer = viewer.launch_passive(
                self._model, self._data,
                show_left_ui=True,
                show_right_ui=True,
            )
            logger.info('MuJoCo viewer launched.')
            return self._viewer
        except Exception as e:
            logger.warning(f'Could not launch viewer: {e}')
            return None

    def sync_viewer(self) -> bool:
        """
        Sync the viewer with current simulation state.

        Returns True if the viewer is still open.
        """
        if self._viewer is None:
            return True
        if not self._viewer.is_running():
            return False
        self._viewer.sync()
        return True

    def close_viewer(self) -> None:
        """Close the viewer."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            logger.info('Viewer closed.')

    def render_offscreen(self, width: int = 640, height: int = 480,
                         camera: int = -1) -> np.ndarray:
        """
        Render a single frame offscreen (useful for headless / logging).

        Returns an RGB numpy array of shape `(height, width, 3)`.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, width, height)

        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render()

    # ---- convenience ------------------------------------------------------

    def close(self) -> None:
        """Clean up all resources."""
        self.close_viewer()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def is_viewer_running(self) -> bool:
        """Check whether the interactive viewer is still open."""
        if self._viewer is None:
            return False
        return self._viewer.is_running()

