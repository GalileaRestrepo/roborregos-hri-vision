import os
from glob import glob
from setuptools import setup, find_packages

setup(
    name='hri_fusion',
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/hri_fusion/launch', glob('launch/*.launch.py')),
        ('share/ament_index/resource_index/packages', ['resource/hri_fusion']),
        ('share/hri_fusion', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='gali',
    author_email='gali@example.com',
    description='Fusion de identidad y politica de decision',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lip_movement_node = hri_fusion.lip_movement_node:main',
            'identity_fusion_node = hri_fusion.identity_fusion_node:main',
            'decision_node = hri_fusion.decision_node:main',
        ],
    },
)
