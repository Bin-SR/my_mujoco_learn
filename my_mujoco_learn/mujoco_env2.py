#!/usr/bin/env python3
"""
MuJoCo environment wrapper for Franka Emika Panda robot.

Loads the Panda model from mujoco_menagerie and provides a clean
interface for simulation stepping, state reading, and rendering.
"""

import os
import re
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
    env_path = os.getenv("MUJOCO_MENAGERIE_PATH")
    if env_path:
        return Path(env_path)
    return Path.home() / "mujoco_menagerie"
    # try:
    #     import mujoco_menagerie
    #     # site-packages/mujoco_menagerie/__init__.py -> package dir
    #     return Path(mujoco_menagerie.__file__).parent
    # except ImportError:
    #     pass

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


def _resolve_include_paths(xml_string: str, menagerie_path: Path) -> str:
    """
    Rewrite <include file="franka_emika_panda/..."/> so that the file
    attribute becomes an absolute path rooted at menagerie_path.

    We deliberately DO NOT touch <compiler>.  This way menagerie's own
    panda.xml (which declares e.g. meshdir="assets") can resolve mesh
    and texture files relative to its own directory — exactly as the
    menagerie authors intended.
    """
    panda_dir = menagerie_path / 'franka_emika_panda'
    if not panda_dir.exists():
        logger.warning(f'franka_emika_panda not found under {menagerie_path}')
        return xml_string

    def _replace_include(match: re.Match) -> str:
        file_val = match.group(1)
        # Only rewrite relative paths pointing into franka_emika_panda/
        if file_val.startswith('franka_emika_panda/') or file_val.startswith('franka_emika_panda\\'):
            abs_path = str(menagerie_path / file_val)
            logger.info(f'Resolved include: {file_val} -> {abs_path}')
            return f'<include file="{abs_path}"/>'
        return match.group(0)
    
    logger.warning(f'*********_replace_include={_replace_include}')
    xml_string = re.sub(
        r'<include\s+file="([^"]*franka_emika_panda[^"]*)"\s*/>',
        _replace_include,
        xml_string,
        flags=re.IGNORECASE,
    )
    return xml_string


def _load_model_xml(xml_path: str,
                    menagerie_path: Optional[Path] = None) -> mujoco.MjModel:
    """
    Load a MuJoCo model from an XML file.

    If menagerie_path is available, <include> paths pointing into
    franka_emika_panda/ are rewritten to absolute paths so that
    menagerie's internal meshdir/texturedir declarations work correctly.
    """
    xml_string = Path(xml_path).read_text(encoding='utf-8')
    logger.warning(f'*********xml_string={xml_string}')
    logger.warning(f'*********xml_path={xml_path}')
    logger.warning(f'*********menagerie_path={menagerie_path}')
    mp = menagerie_path or _find_menagerie_path()
    logger.warning(f'*********menagerie_path22222={menagerie_path}')
    if mp is not None:
        xml_string = _resolve_include_paths(xml_string, mp)
        test = xml_string
    if test == xml_string:
        logger.warning(f'compare same**+-+-+-+-+-+++++++++++////////')
    logger.warning(f'*********xml_string22222={xml_string}')
    return mujoco.MjModel.from_xml_string(xml_string)


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
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 menagerie_path: Optional[str] = None,
                 local_models: Optional[str] = None,
                 render_mode: str = 'window',
                 timestep: float = 0.002,
                 headless: bool = False):
        self._timestep = timestep
        self._render_mode = render_mode
        self._headless = headless

        mp = Path(menagerie_path) if menagerie_path else _find_menagerie_path()

        if model_path is not None:
            xml_path = model_path
        elif local_models is not None:
            xml_path = str(Path(local_models) / 'panda_scene.xml')
        elif mp is not None:
            xml_path = str(mp / 'franka_emika_panda' / 'scene.xml')
        else:
            local = Path(__file__).resolve().parents[1] / 'models' / 'panda_scene.xml'
            if local.exists():
                xml_path = str(local)
            else:
                raise FileNotFoundError(
                    'Could not find Franka Emika Panda model.\n'
                    '  Install:  pip install mujoco-menagerie\n'
                    '  Or clone: git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie\n'
                    '  Or set:   export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie'
                )

        logger.info(f'Loading MuJoCo model from: {xml_path}')
        self._model = _load_model_xml(xml_path, menagerie_path=mp)
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
    # warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

        found = [j for j in PANDA_ALL_JOINTS if j in self.joint_name_to_id]
        if found:
            logger.info(f'Found {len(found)}/{len(PANDA_ALL_JOINTS)} expected Panda joints')

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

    def step(self, ctrl: Optional[np.ndarray] = None) -> None:
        if ctrl is not None:
            self._data.ctrl[:] = ctrl
        else:
            self._data.ctrl[:] = 0.0
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

    def step_n(self, n: int, ctrl: Optional[np.ndarray] = None) -> None:
        for _ in range(n):
            self.step(ctrl)

    def reset(self) -> None:
        mujoco.mj_resetData(self._model, self._data)
        self._step_count = 0
        self._start_time = time.time()

    def get_qpos(self) -> np.ndarray:
        return self._data.qpos.copy()

    def get_qvel(self) -> np.ndarray:
        return self._data.qvel.copy()

    def get_joint_positions(self, joint_names: Optional[list] = None) -> Dict[str, float]:
        names = joint_names or list(self.joint_name_to_id.keys())
        result = {}
        for name in names:
            jid = self.joint_name_to_id.get(name)
            if jid is not None:
                qpos_addr = self._model.jnt_qposadr[jid]
                result[name] = float(self._data.qpos[qpos_addr])
        return result

    def get_joint_velocities(self, joint_names: Optional[list] = None) -> Dict[str, float]:
        names = joint_names or list(self.joint_name_to_id.keys())
        result = {}
        for name in names:
            jid = self.joint_name_to_id.get(name)
            if jid is not None:
                dof_addr = self._model.jnt_dofadr[jid]
                result[name] = float(self._data.qvel[dof_addr])
        return result

    def get_body_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f'Body "{body_name}" not found in model.')
        return self._data.xpos[bid].copy(), self._data.xquat[bid].copy()

    def get_site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise ValueError(f'Site "{site_name}" not found in model.')
        pos = self._data.site_xpos[sid].copy()
        rot = self._data.site_xmat[sid].copy().reshape(3, 3)
        return pos, rot

    def set_control(self, ctrl: np.ndarray) -> None:
        self._data.ctrl[:] = ctrl

    def get_actuator_force(self) -> np.ndarray:
        return self._data.actuator_force.copy()

    def launch_viewer(self) -> Optional[viewer.Handle]:
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
        if self._viewer is None:
            return True
        if not self._viewer.is_running():
            return False
        self._viewer.sync()
        return True

    def close_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            logger.info('Viewer closed.')

    def render_offscreen(self, width: int = 640, height: int = 480,
                         camera: int = -1) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, width, height)
        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        self.close_viewer()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def is_viewer_running(self) -> bool:
        if self._viewer is None:
            return False
        return self._viewer.is_running()
