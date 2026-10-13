from setuptools import setup

setup(
    name="rtisslam_ros",
    version="0.1.0",
    packages=["rtisslam_ros"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/rtisslam_ros"]),
                ("share/rtisslam_ros", ["package.xml"])],
    install_requires=["setuptools"],
    entry_points={"console_scripts": [
        "rtis_sim_node = rtisslam_ros.sim_node:main",
        "rtis_imaging_node = rtisslam_ros.imaging_node:main",
        "rtisslam_node = rtisslam_ros.slam_node:main",
        "rtis_navigator_node = rtisslam_ros.navigator_node:main",
        "rtis_energyscape_viz_node = rtisslam_ros.energyscape_viz_node:main",
    ]},
)
