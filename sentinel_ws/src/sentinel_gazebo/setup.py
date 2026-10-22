import os
from setuptools import setup

package_name = 'sentinel_gazebo'


def tree(src):
    out = []
    for root, _, files in os.walk(src):
        if files:
            out.append((os.path.join('share', package_name, root),
                        [os.path.join(root, f) for f in files]))
    return out


setup(
    name=package_name,
    version='2.0.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])]
               + tree('models') + tree('worlds') + tree('scripts'),
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Karthik KS',
    maintainer_email='karthikzofficialmail@gmail.com',
    description='Gazebo assets for SENTINEL.',
    license='MIT',
)
