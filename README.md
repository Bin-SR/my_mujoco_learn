# My MuJoCo Learning

A personal project for learning MuJoCo physics simulation with ROS2 integration.

## mujoco_panda

ROS2 Humble package that runs a MuJoCo simulation of the Franka Emika Panda robotic arm.

### Features

- **MuJoCo physics simulation** with the Franka Emika Panda (from mujoco_menagerie)
- **ROS2 integration**: publishes `/joint_states` and TF transforms
- **Interactive viewer**: MuJoCo passive viewer for real-time visualization
- **Ready for MoveIt + RViz**: joint states and TF are published for downstream tools
- **Extensible**: clean environment wrapper for building RL / embodied AI pipelines

### Requirements

- Ubuntu 22.04
- ROS2 Humble
- MuJoCo (Python bindings)
- mujoco_menagerie (`pip install mujoco-menagerie`)

### Quick Start

```bash
# 1. Install dependencies
pip install mujoco mujoco-menagerie

# 2. Clone & build
cd ~/ros2_ws/src
git clone https://github.com/Bin-SR/my_mujoco_learn.git
cd ~/ros2_ws
colcon build --packages-select mujoco_panda
source install/setup.bash

# 3. Launch simulation
ros2 launch mujoco_panda panda_sim.launch.py

# 4. (Optional) Launch without viewer for RViz/MoveIt
ros2 launch mujoco_panda panda_sim.launch.py use_viewer:=false
```

### Package Structure

```
mujoco_panda/
├── mujoco_panda/           # Python package
│   ├── __init__.py
│   ├── mujoco_env.py       # MuJoCo environment wrapper
│   └── panda_sim_node.py   # ROS2 simulation node
├── launch/
│   └── panda_sim.launch.py # Launch file
├── config/
│   └── panda_sim.yaml      # Default parameters
├── models/
│   └── panda_scene.xml     # Local fallback scene
├── scripts/
│   ├── download_menagerie.sh  # Download menagerie models
│   └── setup_models.py        # Copy models from pip package
├── package.xml
├── setup.py
├── setup.cfg
└── CMakeLists.txt
```

### Topics

| Topic             | Type                          | Direction |
| ----------------- | ----------------------------- | --------- |
| `/joint_states`   | `sensor_msgs/JointState`      | publish   |
| `/tf`             | `tf2_msgs/TFMessage`          | publish   |
| `/joint_commands` | `std_msgs/Float64MultiArray`  | subscribe |

### Next Steps (Roadmap)

- [ ] Generate URDF from MuJoCo MJCF for `robot_state_publisher` + RViz
- [ ] Integrate MoveIt2 for motion planning with Panda
- [ ] Add camera sensor simulation and image publishing
- [ ] Build pick-and-place task environment
- [ ] Implement RL training loop (gymnasium-compatible)
- [ ] Add multi-robot / tabletop scenarios
