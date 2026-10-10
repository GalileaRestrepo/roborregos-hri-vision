from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hri_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agrm',
    maintainer_email='lofiag1296@gmail.com',
    description='Camera, face detection, tracking and visualization nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = hri_vision.camera_node:main',
            'face_detection_node = hri_vision.face_detection_node:main',
            'face_tracker_node = hri_vision.face_tracker_node:main',
            'enroll_action_server = hri_vision.enroll_action_server:main',
        ],
    },
)
