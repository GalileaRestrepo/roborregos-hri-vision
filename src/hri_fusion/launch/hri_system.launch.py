"""
launch file principal del sistema HRI/Vision
uso: ros2 launch hri_fusion hri_system.launch.py

nota: en macOS los nodos de vision y audio necesitan correr en el proceso
principal para acceder a camara y microfono. por eso usamos ExecuteProcess
en lugar de Node() para esos componentes.
"""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

PYTHON    = '/Users/agrm/miniconda3/envs/ros2_hri/bin/python'
WS        = os.path.expanduser('~/ros2_hri_ws')
PYPATH    = ':'.join([
    f'{WS}/install/hri_fusion/lib/python3.9/site-packages',
    f'{WS}/install/hri_vision/lib/python3.9/site-packages',
    f'{WS}/install/hri_audio/lib/python3.9/site-packages',
    f'{WS}/install/hri_interfaces/lib/python3.9/site-packages',
    f'{WS}/install/hri_rag/lib/python3.9/site-packages',
])
AMENT = f'{WS}/install:/Users/agrm/miniconda3/envs/ros2_hri'
ENV   = {'PYTHONPATH': PYPATH, 'AMENT_PREFIX_PATH': AMENT}


def generate_launch_description():
    # mcp_server (FastAPI - no es nodo ROS)
    mcp_server = ExecuteProcess(
        cmd=[PYTHON, '-m', 'hri_rag.mcp_server'],
        additional_env=ENV,
        output='screen',
        name='mcp_server'
    )

    # rag_service_node (servicio ROS 2)
    rag_service = Node(
        package='hri_rag',
        executable='rag_service_node',
        name='rag_service_node',
        output='screen',
        additional_env=ENV
    )

    # nodos de fusion (no necesitan camara ni mic, funcionan como subprocesos)
    identity_fusion = Node(
        package='hri_fusion',
        executable='identity_fusion_node',
        name='identity_fusion_node',
        output='screen',
        additional_env=ENV
    )

    decision = Node(
        package='hri_fusion',
        executable='decision_node',
        name='decision_node',
        output='screen',
        additional_env=ENV
    )

    # sistema principal: vision + audio + GUI en un solo proceso foreground
    # (necesario en macOS para acceso a camara, microfono y ventana)
    hri_system = TimerAction(
        period=3.0,  # espera 3s a que rag_service y mcp_server arranquen
        actions=[
            ExecuteProcess(
                cmd=[PYTHON, f'{WS}/scripts/hri_system.py'],
                additional_env=ENV,
                output='screen',
                name='hri_system'
            )
        ]
    )

    return LaunchDescription([
        mcp_server,
        rag_service,
        identity_fusion,
        decision,
        hri_system,
    ])
