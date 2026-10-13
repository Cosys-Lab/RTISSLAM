"""RTISSLAM in the closed-loop simulator (everything on sim time from rtis_sim_node).

    ros2 launch rtisslam_bringup sim.launch.py world:=office mode:=mapping
    ros2 launch rtisslam_bringup sim.launch.py world:=office mode:=localization map_file:=/path/map.npz \
        start:=4.84,10.95,-0.77 use_start:=true

mapping: drive with teleop (or any cmd_vel source), then `ros2 service call /rtisslam/save_map std_srvs/srv/Trigger`.
localization: RTISSLAM localises in the map and publishes /map for Nav2; send goals with RViz "2D Goal Pose".
Environment: RTISSLAM_ROOT (the project folder), RTISSLAM_DATA (models, maps).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("rtisslam_bringup")
    nav2 = os.path.join(share, "config", "nav2_planner.yaml")
    sim = {"use_sim_time": True}
    start = ParameterValue(LC("start"), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="office"),
        DeclareLaunchArgument("mode", default_value="mapping"),
        DeclareLaunchArgument("map_file", default_value=""),
        DeclareLaunchArgument("save_file", default_value="rtisslam_map.npz"),
        DeclareLaunchArgument("start", default_value="0.0,0.0,0.0"),
        DeclareLaunchArgument("use_start", default_value="false"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("real_time_factor", default_value="0.0"),
        Node(package="rtisslam_ros", executable="rtis_sim_node", name="rtis_sim", output="screen",
             parameters=[{"world": LC("world"), "device": LC("device"), "methods": ["deconv:ams15", "deconv:none", "dmas3cf:ams2s", "deconv"], "start": start,
                          "use_start": ParameterValue(LC("use_start"), value_type=bool),
                          "real_time_factor": ParameterValue(LC("real_time_factor"), value_type=float)}]),
        Node(package="rtisslam_ros", executable="rtisslam_node", name="rtisslam", output="screen",
             parameters=[sim, {"mode": LC("mode"), "map_file": LC("map_file"), "save_file": LC("save_file"),
                               "start": start}]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server", output="screen",
             parameters=[nav2, sim]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_planning",
             output="screen", parameters=[sim, {"autostart": True, "node_names": ["planner_server"]}]),
        Node(package="rtisslam_ros", executable="rtis_navigator_node", name="rtis_navigator", output="screen",
             parameters=[sim]),
    ])
