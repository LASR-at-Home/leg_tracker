import launch
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription(
        [
            launch.actions.DeclareLaunchArgument(
                "forest_file",
                default_value="trained_leg_detector_res=0.33.yaml",
                description="Path to trained leg detector config file",
            ),
            launch.actions.DeclareLaunchArgument(
                "scan_topic", default_value="/scan", description="Laser scan topic"
            ),
            launch.actions.DeclareLaunchArgument(
                "fixed_frame",
                default_value="/base_link",
                description="Fixed frame reference",
            ),
            launch.actions.DeclareLaunchArgument(
                "scan_frequency", default_value="10", description="Scan frequency"
            ),
            launch_ros.actions.Node(
                package="leg_tracker",
                executable="detect_leg_clusters",
                name="detect_leg_clusters",
                output="screen",
                parameters=[
                    {
                        "forest_file": launch.substitutions.LaunchConfiguration(
                            "forest_file"
                        ),
                        "scan_topic": launch.substitutions.LaunchConfiguration(
                            "scan_topic"
                        ),
                        "fixed_frame": launch.substitutions.LaunchConfiguration(
                            "fixed_frame"
                        ),
                        "scan_frequency": launch.substitutions.LaunchConfiguration(
                            "scan_frequency"
                        ),
                    }
                ],
            ),
            launch_ros.actions.Node(
                package="leg_tracker",
                executable="joint_leg_tracker.py",
                name="joint_leg_tracker",
                output="screen",
            ),
            launch_ros.actions.Node(
                package="leg_tracker",
                executable="local_occupancy_grid_mapping",
                name="local_occupancy_grid_mapping",
                output="screen",
            ),
        ]
    )
