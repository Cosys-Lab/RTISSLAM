"""Interactive demo: the closed-loop eRTIS simulator, RTISSLAM localizing in a saved map, Nav2 global planning, the
acoustic-flow navigator, and RViz2 with the occupancy grid, the pose graph, the energyscapes and a goal tool.

    ros2 launch rtisslam_bringup demo.launch.py world:=office map_file:=$RTISSLAM_DATA/runs/office/map.npz

Give the robot a goal with the "2D Goal Pose" tool in RViz2. The start pose of each world is the start of its mapping
drive (set with start:=x,y,yaw for another map). methods lists the energyscapes the simulator publishes: those the map
was built with ("imaging:post" keys, e.g. dmas3cf:ams2s) plus the controller's afc_method; viz_method is the one shown.
Environment: RTISSLAM_ROOT (the project folder), RTISSLAM_DATA (models, maps), RTISSLAM_VENV (the Python venv).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node

STARTS = {"office": "4.839,10.952,-0.772", "warehouse": "3.168,2.164,-0.493", "atrium": "21.438,16.551,-3.110"}


def nodes(context):
    share = get_package_share_directory("rtisslam_bringup")
    nav2 = os.path.join(share, "config", "nav2_planner.yaml")
    rviz = LC("rviz_config").perform(context) or os.path.join(share, "config", "rtisslam.rviz")
    world = LC("world").perform(context)
    start = LC("start").perform(context) or STARTS.get(world, "0,0,0")
    map_file = LC("map_file").perform(context) or os.path.join(
        os.environ.get("RTISSLAM_DATA", os.path.expanduser("~/RTISSLAM_data")), "runs", world, "map.npz")
    methods = [m.strip() for m in LC("methods").perform(context).split(",") if m.strip()]
    # raw:=true runs the real-sensor data path: the simulator publishes raw microphone signals (RtisPulse) and
    # rtis_imaging_node turns them into energyscapes, as on the robot
    raw = LC("raw").perform(context).lower() in ("1", "true", "yes")
    sim = {"use_sim_time": True}
    out = [
        Node(package="rtisslam_ros", executable="rtis_sim_node", name="rtis_sim", output="screen",
             parameters=[{"world": world, "device": LC("device").perform(context), "methods": methods, "start": start,
                          "use_start": True, "real_time_factor": float(LC("real_time_factor").perform(context)),
                          "output": "pulse" if raw else "energyscape"}]),
        Node(package="rtisslam_ros", executable="rtisslam_node", name="rtisslam", output="screen",
             parameters=[sim, {"mode": "localization", "map_file": map_file, "start": start,
                               "localizer": LC("localizer").perform(context)}]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server", output="screen",
             parameters=[nav2, sim]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_planning",
             output="screen", parameters=[sim, {"autostart": True, "node_names": ["planner_server"]}]),
        Node(package="rtisslam_ros", executable="rtis_navigator_node", name="rtis_navigator", output="screen",
             parameters=[sim, {"afc_method": LC("afc_method").perform(context)}]),
        Node(package="rtisslam_ros", executable="rtis_energyscape_viz_node", name="rtis_energyscape_viz",
             output="screen", parameters=[sim, {"method": LC("viz_method").perform(context)}]),
    ]
    if raw:
        out.append(Node(package="rtisslam_ros", executable="rtis_imaging_node", name="rtis_imaging", output="screen",
                        parameters=[sim, {"methods": methods, "device": LC("device").perform(context)}]))
    if LC("rviz").perform(context).lower() in ("1", "true", "yes"):
        out.append(Node(package="rviz2", executable="rviz2", name="rviz2", arguments=["-d", rviz],
                        parameters=[sim], output="log"))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="office"),
        DeclareLaunchArgument("map_file", default_value=""),
        DeclareLaunchArgument("start", default_value=""),
        DeclareLaunchArgument("methods", default_value="dmas3cf:ams2s"),
        DeclareLaunchArgument("afc_method", default_value="dmas3cf:ams2s",
                              description="energyscape of the acoustic-flow controller (must be in methods)"),
        DeclareLaunchArgument("viz_method", default_value="dmas3cf:ams2s"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("real_time_factor", default_value="1.0"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value="", description="RViz2 config (default config/rtisslam.rviz)"),
        DeclareLaunchArgument("raw", default_value="false", description="real-sensor data path (RtisPulse + imaging node)"),
        DeclareLaunchArgument("localizer", default_value="pf", description="pf | mix | graph (rtisslam/localize.py); pf is best in closed loop"),
        OpaqueFunction(function=nodes),
    ])
