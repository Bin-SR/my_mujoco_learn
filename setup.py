from setuptools import find_packages, setup

package_name = 'mujoco_learn'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/panda_sim.launch.py']),
        ('share/' + package_name + '/config', ['config/panda_sim.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Bin_SR',
    maintainer_email='1072235132@qq.com',
    description='MuJoCo simulation of Franka Emika Panda arm with ROS2 integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'panda_sim_node = mujoco_learn.panda_sim_node:main',
        ],
    },
)
