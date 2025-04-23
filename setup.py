from setuptools import setup, find_packages

package_name = 'robotic_arm_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['robotic_arm_planner', 'robotic_arm_planner.*']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yeray',
    maintainer_email='ynavarro@cdei.upc.edu',
    description='Planner node for UR10e robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = robotic_arm_planner.planner_node:main',
        ],
    },
)
