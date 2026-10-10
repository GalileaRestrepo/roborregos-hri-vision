from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # 1. abre la cam y publica frames
        Node(
            package='hri_vision',
            executable='camera_node',
            name='camera_node',
            output='screen',
        ),

        # 2. detecta caras en cada frame (insightface)
        Node(
            package='hri_vision',
            executable='face_detection_node',
            name='face_detection_node',
            output='screen',
        ),

        # 3. da ids estables e identifica quien es cada cara
        Node(
            package='hri_vision',
            executable='face_tracker_node',
            name='face_tracker_node',
            output='screen',
        ),

        # 4. dibuja las cajas y nombres en pantalla
        Node(
            package='hri_vision',
            executable='visualization_node',
            name='visualization_node',
            output='screen',
        ),
    ])
