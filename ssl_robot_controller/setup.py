from setuptools import find_packages, setup

package_name = 'ssl_robot_controller'

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
    maintainer='muhtashim',
    maintainer_email='muhtashim@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'teleop_node = ssl_robot_controller.teleop_node:main',
            'tracker_node = ssl_robot_controller.tracker_node:main',
            'navigator_node = ssl_robot_controller.navigator_node:main',
            'autonomous_node = ssl_robot_controller.autonomous_node:main',
            'dynamic_robot = ssl_robot_controller.dynamic_robot:main',
        ],
    },
)
