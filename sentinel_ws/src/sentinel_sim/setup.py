from glob import glob
from setuptools import find_packages, setup

package_name = 'sentinel_sim'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/configs', glob('configs/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'shapely', 'pandas', 'pyyaml'],
    zip_safe=True,
    maintainer='Karthik KS',
    maintainer_email='karthikzofficialmail@gmail.com',
    description='Fast multi-UAV simulator for SENTINEL.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': ['sentinel_run = sentinel_sim.runner:main']},
)
