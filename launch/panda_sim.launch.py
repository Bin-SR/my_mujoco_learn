#!/usr/bin/env python3
'''
Launch file for the MuJoCo Panda simulation.

Starts:
  - panda_sim_node      (MuJoCo physics + joint state / TF publishing)
  - robot_state_publisher (optional, reads URDF for RViz)
  - joint_state_publisher_gui (optional, manual joint sliders)

Usage::

    ros2 launch mujoco_learn panda_sim.launch.py

    # Without viewer (headless, for MoveIt / RViz only):
    ros2 launch mujoco_learn panda_sim.launch.py use_viewer:=false

    # With a specific menagerie path:
    ros2 launch mujoco_learn panda_sim.launch.py menagerie_path:=/opt/mujoco_menagerie
'''

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    RegisterEventHandler,
    LogInfo,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():
    # ── package paths ──────────────────────────────────────────────────
    pkg_share = get_package_share_directory('my_mujoco_learn')
    # /home/ub22/ros2_ws/install/my_mujoco_learn/share/my_mujoco_learn
    print(pkg_share)

    # ── launch arguments ───────────────────────────────────────────────
    # 这里的model_path和menagerie_path_arg, 默认值为''，即空字符串，为none
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='',
        description='Direct path to Panda XML/MJCF file (overrides auto-detect).',
    )
    menagerie_path_arg = DeclareLaunchArgument(
        'menagerie_path', default_value='',
        description='Path to mujoco_menagerie root directory.',
    )
    timestep_arg = DeclareLaunchArgument(
        'timestep', default_value='0.002',
        description='Simulation timestep (seconds).',
    )
    rate_arg = DeclareLaunchArgument(
        'rate', default_value='500',
        description='ROS timer rate (Hz) for physics stepping.',
    )
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run without any rendering (headless server mode).',
    )
    use_viewer_arg = DeclareLaunchArgument(
        'use_viewer', default_value='true',
        description='Launch the MuJoCo interactive viewer.',
    )

    # ── environment ────────────────────────────────────────────────────
    # Help mujoco_learn nodes find bundled resources
    set_share_env = SetEnvironmentVariable(
        name='mujoco_learn_SHARE',
        value=pkg_share,
    )

    # ── simulation node ────────────────────────────────────────────────
    panda_sim_node = Node(
        package='my_mujoco_learn',
        executable='panda_sim_node',
        name='panda_sim_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'menagerie_path': LaunchConfiguration('menagerie_path'),
            'local_models': os.path.join(pkg_share, 'models'),
            'timestep': LaunchConfiguration('timestep'),
            'rate': LaunchConfiguration('rate'),
            'headless': LaunchConfiguration('headless'),
            'use_viewer': LaunchConfiguration('use_viewer'),
            'publish_tf': True,
        }],
        # Auto-restart on crash
        respawn=False,
    )

    # ── optional: robot_state_publisher (for RViz) ─────────────────────
    # If you provide a Panda URDF in the urdf/ directory, uncomment this:
    #
    # robot_state_publisher_node = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     parameters=[{
    #         'robot_description': PathJoinSubstitution([
    #             pkg_share, 'urdf', 'panda.urdf'
    #         ]),
    #     }],
    # )

    # ── optional: joint_state_publisher_gui ────────────────────────────
    # joint_state_publisher_gui_node = Node(
    #     package='joint_state_publisher_gui',
    #     executable='joint_state_publisher_gui',
    #     name='joint_state_publisher_gui',
    # )

    # ── assemble ───────────────────────────────────────────────────────
    ld = LaunchDescription([
        model_path_arg,
        menagerie_path_arg,
        timestep_arg,
        rate_arg,
        headless_arg,
        use_viewer_arg,
        set_share_env,
        panda_sim_node,
        # robot_state_publisher_node,
        # joint_state_publisher_gui_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=panda_sim_node,
                on_exit=[LogInfo(msg='panda_sim_node exited. Shutting down.')],
            )
        ),
    ])

    return ld
