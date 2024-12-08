from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # realsense node
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera_node',
        output='screen',
        parameters=[{'align_depth.enable': True},
                    {'pointcloud.enable': True}
                    # {'color_width':640},
                    # {'color_height':360},
                    # {'depth_width':640},
                    # {'depth_height':360},
                    ]
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
    # launch tracker node
    tracker_node = Node(
        package='realsense',
        executable='tracker_node',
        name='tracker',
        output='screen'
    )

    # launch depth node
    depth_node = Node(
        package='realsense',
        executable='depth_node',
        name='depth_processor',
        output='screen'
    )

    #launch rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([realsense_node, image_processor_node, detector_node, tracker_node, depth_node, rviz_node])