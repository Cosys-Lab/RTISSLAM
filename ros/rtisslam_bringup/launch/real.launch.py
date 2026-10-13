"""RTISSLAM on the real robot: Pioneer (differential drive) + eRTIS + a lidar for ground truth.

Differs from sim.launch.py only in the sensor and robot nodes:
  robot      the Pioneer driver (an existing ROS 2 driver, e.g. AriaCoda + ros2aria), included through the
             `robot_launch` argument: publishes odom + TF odom -> base_link, subscribes cmd_vel
  eRTIS      the eRTIS ROS 2 driver (`ertis_launch`) + an adapter to rtis_msgs/RtisPulse, then rtis_imaging_node
             (DMAS3 + coherence factor; GPU for 10 Hz)
  lidar      the lidar driver (`lidar_launch`) and slam_toolbox in its own map frame with TF publishing OFF
             (transform_publish_period 0): its /pose is the REFERENCE trajectory, recorded in the bag next to
             RTISSLAM's estimate; it never feeds RTISSLAM or the navigation
Real time (use_sim_time false). Static TF base_link -> rtis_link (the eRTIS mount; measure it on the robot).

    ros2 launch rtisslam_bringup real.launch.py mode:=mapping robot_launch:=/path/pioneer.launch.py \
        ertis_launch:=/path/ertis.launch.py lidar_launch:=/path/lidar.launch.py
    ros2 launch rtisslam_bringup real.launch.py mode:=localization map_file:=/path/map.npz localizer:=pf \
        start:=x,y,yaw imaging_device:=cuda ...      (start "" with localizer mix / pf: unknown start, turn in place)
Imaging needs a GPU for DMAS3-CF at 10 Hz (CPU: ~0.6 s per pulse); see report/tonight, section "Toward the Real Robot".
    ros2 bag record /odom /tf /tf_static /rtis/energyscape /rtis/pulse /rtisslam/pose /pose /scan /cmd_vel /rtis_nav/layer
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _include(arg):
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(LC(arg)),
                                    condition=IfCondition(PythonExpression(['"', LC(arg), '" != ""'])))


def _imaging(context):
    methods = [m.strip() for m in LC("methods").perform(context).split(",") if m.strip()]
    return [Node(package="rtisslam_ros", executable="rtis_imaging_node", name="rtis_imaging", output="screen",
                 parameters=[{"use_sim_time": False, "methods": methods, "device": LC("imaging_device").perform(context)}])]


def generate_launch_description():
    share = get_package_share_directory("rtisslam_bringup")
    nav2 = os.path.join(share, "config", "nav2_planner.yaml")
    real = {"use_sim_time": False}
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="mapping"),
        DeclareLaunchArgument("map_file", default_value=""),
        DeclareLaunchArgument("save_file", default_value="rtisslam_map.npz"),
        DeclareLaunchArgument("start", default_value="0.0,0.0,0.0"),
        DeclareLaunchArgument("robot_launch", default_value=""),
        DeclareLaunchArgument("ertis_launch", default_value=""),
        DeclareLaunchArgument("lidar_launch", default_value=""),
        DeclareLaunchArgument("ertis_x", default_value="0.10"),
        DeclareLaunchArgument("ertis_z", default_value="0.40"),
        DeclareLaunchArgument("localizer", default_value="pf", description="localization: pf | mix | graph (pf is best in closed loop)"),
        DeclareLaunchArgument("imaging_device", default_value="cpu", description="cpu | cuda (DMAS3-CF needs cuda for 10 Hz)"),
        DeclareLaunchArgument("methods", default_value="dmas3cf:ams2s",
                              description="energyscape keys to image: those the map uses (+ the controller's afc_method)"),
        DeclareLaunchArgument("afc_method", default_value="dmas3cf:ams2s",
                              description="energyscape of the acoustic-flow controller (must be in methods)"),
        _include("robot_launch"), _include("ertis_launch"), _include("lidar_launch"),
        Node(package="tf2_ros", executable="static_transform_publisher", name="rtis_mount",
             arguments=["--x", LC("ertis_x"), "--z", LC("ertis_z"), "--frame-id", "base_link",
                        "--child-frame-id", "rtis_link"]),
        OpaqueFunction(function=_imaging),
        Node(package="rtisslam_ros", executable="rtisslam_node", name="rtisslam", output="screen",
             parameters=[real, {"mode": LC("mode"), "map_file": LC("map_file"), "save_file": LC("save_file"),
                                "start": ParameterValue(LC("start"), value_type=str), "localizer": LC("localizer")}]),
        Node(package="slam_toolbox", executable="async_slam_toolbox_node", name="lidar_reference", output="screen",
             parameters=[real, {"map_frame": "lidar_map", "odom_frame": "odom", "base_frame": "base_link",
                                "transform_publish_period": 0.0, "mode": "mapping"}],
             remappings=[("/map", "/lidar_reference/map")]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server", output="screen",
             parameters=[nav2, real]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_planning",
             output="screen", parameters=[real, {"autostart": True, "node_names": ["planner_server"]}]),
        Node(package="rtisslam_ros", executable="rtis_navigator_node", name="rtis_navigator", output="screen",
             parameters=[real, {"afc_method": LC("afc_method")}]),
    ])
