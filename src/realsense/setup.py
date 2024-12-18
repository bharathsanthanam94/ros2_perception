from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'realsense'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bharath',
    maintainer_email='bharathsanthanamd1@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "rs_node = realsense.realsense_node:main",
            "detector_node = realsense.detector_node:main",
            "tracker_node = realsense.tracker_node:main",
            "depth_node = realsense.depth_node:main"
        ],
    },
)
