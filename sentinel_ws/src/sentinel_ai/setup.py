import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sentinel_ai'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Karthik KS',
    maintainer_email='karthikzofficialmail@gmail.com',
    description='SENTINEL ROS 2 nodes: perception, decentralised EM-PSO agent, PX4 offboard, '
                'channel emulation, failure injection and metrics.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'thermal_vision = sentinel_ai.thermal_vision:main',
            'flight_commander = sentinel_ai.flight_commander:main',
            'emps_agent = sentinel_ai.emps_agent:main',
            'comm_emulator = sentinel_ai.comm_emulator:main',
            'chaos_monkey = sentinel_ai.chaos_monkey:main',
            'metrics_logger = sentinel_ai.metrics_logger:main',
            'sim_detector = sentinel_ai.sim_detector:main',
        ],
    },
)
