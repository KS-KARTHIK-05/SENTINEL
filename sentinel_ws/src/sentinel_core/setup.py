from setuptools import find_packages, setup

package_name = 'sentinel_core'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Karthik KS',
    maintainer_email='karthikzofficialmail@gmail.com',
    description='ROS-independent SENTINEL algorithms.',
    license='MIT',
    extras_require={'test': ['pytest']},
)
