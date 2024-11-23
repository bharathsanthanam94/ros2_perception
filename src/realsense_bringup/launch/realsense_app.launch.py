from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # realsense node
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera_node',
        output='screen'
    )

    # launch image publishing node
    image_processor_node = Node(
        package='realsense',
        executable='rs_node',
        name='realsense_processor',
        output='screen'
    )

    #launch detector node
    detector_node = Node(
        package='realsense',
        executable='detector_node',
        name='detector',
        output='screen'
    )

    #launch rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([realsense_node, image_processor_node, detector_node, rviz_node])