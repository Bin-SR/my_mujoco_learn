#!/usr/bin/env python3
"""
ROS2 node that runs a MuJoCo simulation of the Franka Emika Panda arm.

Publishes
---------
/joint_states  (sensor_msgs/JointState)
/tf            (via tf2_ros.TransformBroadcaster)

Subscribes
----------
/joint_commands  (std_msgs/Float64MultiArray)   – direct actuator targets
"""

import os
import sys
import signal
import threading
from pathlib import Path
from typing import Optional, Dict, List

import mujoco
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
import tf2_ros

from .mujoco_env import MujocoPandaEnv, PANDA_ALL_JOINTS

# ── constants ──────────────────────────────────────────────────────────────
NODE_NAME = 'panda_sim_node'
DEFAULT_RATE = 500  # Hz – simulation control loop


class PandaSimNode(Node):
    """
    Main ROS2-MuJoCo bridge for the Panda arm.

    Spins the MuJoCo physics at a configurable rate, publishes joint states
    and TF for every body/site of interest, and accepts external joint
    commands.
    """

    def __init__(self):
        super().__init__(NODE_NAME)

        # ── parameters ────────────────────────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('menagerie_path', '')
        self.declare_parameter('local_models', '')
        self.declare_parameter('timestep', 0.002)
        self.declare_parameter('rate', DEFAULT_RATE)
        self.declare_parameter('headless', False)
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('joint_names', PANDA_ALL_JOINTS)

        model_path = self.get_parameter('model_path').value or None
        menagerie_path = self.get_parameter('menagerie_path').value or None
        local_models = self.get_parameter('local_models').value or None

        # Resolve default local_models relative to share directory
        if not local_models:
            share_dir = os.environ.get(
                'MUJOCO_LEARN_SHARE',
                str(Path(__file__).resolve().parents[2] / 'share' / 'mujoco_panda')
            )
            candidate = Path(share_dir) / 'models'
            if candidate.exists():
                local_models = str(candidate)

        timestep = self.get_parameter('timestep').value
        headless = self.get_parameter('headless').value

        # ── init MuJoCo environment ────────────────────────────────────
        self.get_logger().info('Initialising MuJoCo Panda environment ...')
        self._env = MujocoPandaEnv(
            model_path=model_path,
            menagerie_path=menagerie_path,
            local_models=local_models,
            timestep=timestep,
            headless=headless,
        )
        self.get_logger().info(
            f'MuJoCo model loaded. '
            f'nj={self._env.nj}  nu={self._env.nu}  timestep={timestep}s'
        )

        self._joint_names: List[str] = self.get_parameter('joint_names').value
        self.get_logger().info(f'Tracked joints: {self._joint_names}')

        # ── ROS interfaces ─────────────────────────────────────────────
        cbg = ReentrantCallbackGroup()

        self._js_pub = self.create_publisher(
            JointState, '/joint_states', 10,
            callback_group=cbg,
        )

        self._cmd_sub = self.create_subscription(
            Float64MultiArray, '/joint_commands',
            self._cmd_callback, 10,
            callback_group=cbg,
        )

        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._latest_command: Optional[np.ndarray] = None

        # ── timer-driven simulation loop ───────────────────────────────
        rate = self.get_parameter('rate').value
        period = 1.0 / max(rate, 1)
        self._timer = self.create_timer(period, self._simulation_tick)

        # ── viewer (non-blocking, launched outside main thread) ────────
        if self.get_parameter('use_viewer').value and not headless:
            self._viewer_thread = threading.Thread(
                target=self._viewer_loop, daemon=True, name='mujoco-viewer'
            )
            self._viewer_thread.start()
        else:
            self._viewer_thread = None

        self.get_logger().info(f'{NODE_NAME} ready.')

    # ── command callback ───────────────────────────────────────────────────
    def _cmd_callback(self, msg: Float64MultiArray) -> None:
        """Receive joint command targets and buffer them."""
        arr = np.array(msg.data, dtype=np.float64)
        if len(arr) != self._env.nu:
            self.get_logger().warn(
                f'Expected {self._env.nu} commands, got {len(arr)}. Ignoring.',
                throttle_duration_sec=1.0,
            )
            return
        self._latest_command = arr

    # ── main simulation tick ───────────────────────────────────────────────
    def _simulation_tick(self) -> None:
        """Called by ROS timer – advance sim, publish state."""
        # Step physics
        self._env.step(self._latest_command)

        # Publish joint states
        self._publish_joint_states()

        # Publish TF
        if self.get_parameter('publish_tf').value:
            self._publish_tf()

    # ── joint state publishing ─────────────────────────────────────────────
    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._joint_names

        positions = self._env.get_joint_positions(self._joint_names)
        velocities = self._env.get_joint_velocities(self._joint_names)

        for name in self._joint_names:
            msg.position.append(positions.get(name, 0.0))
            msg.velocity.append(velocities.get(name, 0.0))
            msg.effort.append(0.0)

        self._js_pub.publish(msg)

    # ── TF publishing ──────────────────────────────────────────────────────
    def _publish_tf(self) -> None:
        """Publish TF for each body relative to world."""
        now = self.get_clock().now().to_msg()
        model = self._env.model
        nbody = model.nbody

        for i in range(1, nbody):  # skip world body (id 0)
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, i
            )
            if body_name is None:
                continue

            pos = self._env.data.xpos[i]
            quat = self._env.data.xquat[i]  # w, x, y, z

            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'world'
            t.child_frame_id = body_name
            t.transform.translation = Vector3(
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2])
            )
            t.transform.rotation = Quaternion(
                w=float(quat[0]), x=float(quat[1]),
                y=float(quat[2]), z=float(quat[3]),
            )
            self._tf_broadcaster.sendTransform(t)

    # ── viewer (runs in separate thread) ───────────────────────────────────
    def _viewer_loop(self) -> None:
        """Run MuJoCo passive viewer in a background thread."""
        self.get_logger().info('Launching MuJoCo viewer in background thread ...')
        try:
            self._env.launch_viewer()
            while rclpy.ok() and self._env.is_viewer_running():
                self._env.sync_viewer()
                import time
                time.sleep(0.001)
        except Exception as e:
            self.get_logger().error(f'Viewer thread error: {e}')
        finally:
            self._env.close_viewer()
            self.get_logger().info('Viewer thread exiting.')

    # ── shutdown ───────────────────────────────────────────────────────────
    def destroy_node(self):
        self._env.close()
        super().destroy_node()


# ── entry point ────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)

    node = PandaSimNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
