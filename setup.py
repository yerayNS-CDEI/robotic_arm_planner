from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'robotic_arm_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['robotic_arm_planner', 'robotic_arm_planner.*']),
    data_files=[
    	('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','numpy','scipy','matplotlib'],
    zip_safe=True,
    maintainer='Yeray',
    maintainer_email='ynavarro@cdei.upc.edu',
    description='Planner package for a mobile  manipulator. Currnetly using the UR10e robot.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = robotic_arm_planner.planner_node:main',
            'base_placement_node = robotic_arm_planner.base_placement_node:main',
        ],
    },
)
