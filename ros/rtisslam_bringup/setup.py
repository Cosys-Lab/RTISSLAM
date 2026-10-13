from glob import glob

from setuptools import setup

setup(
    name="rtisslam_bringup",
    version="0.1.0",
    packages=[],
    data_files=[("share/ament_index/resource_index/packages", ["resource/rtisslam_bringup"]),
                ("share/rtisslam_bringup", ["package.xml"]),
                ("share/rtisslam_bringup/launch", glob("launch/*.py")),
                ("share/rtisslam_bringup/config", glob("config/*"))],
    install_requires=["setuptools"],
)
