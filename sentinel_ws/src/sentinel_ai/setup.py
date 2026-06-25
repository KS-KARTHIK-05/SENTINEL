import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sentinel_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nicholas',
    maintainer_email='nicholas@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'thermal_vision = sentinel_ai.thermal_vision:main',
            'flight_commander = sentinel_ai.flight_commander:main',
            'swarm_coordinator = sentinel_ai.swarm_coordinator:main',
        ],
    },
)
