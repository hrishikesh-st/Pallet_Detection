from setuptools import find_packages, setup

package_name = 'pallet_detector'

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
    maintainer='megatron',
    maintainer_email='hpawar@wpi.edu',
    description='ROS2 package for pallet detection and segmentation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = pallet_detector.detection_node:main',
            'segmentation_node = pallet_detector.segmentation_node:main',
        ],
    },
)
