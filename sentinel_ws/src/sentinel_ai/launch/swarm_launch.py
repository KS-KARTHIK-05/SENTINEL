from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    ld = LaunchDescription()
    num_drones = 3

    # 1. Boot the Central Hive Mind immediately
    coordinator = Node(
        package='sentinel_ai',
        executable='swarm_coordinator',
        name='swarm_coordinator',
        output='screen'
    )
    ld.add_action(coordinator)

    # 2. Stagger the AI and Flight Controllers
    for i in range(num_drones):
        namespace = f'/px4_{i}'
        
        commander = Node(
            package='sentinel_ai',
            executable='flight_commander',
            name=f'flight_commander_{i}',
            arguments=[namespace],
            output='screen'
        )

        vision = Node(
            package='sentinel_ai',
            executable='thermal_vision',
            name=f'thermal_vision_{i}',
            arguments=[namespace],
            output='screen'
        )

        bridge = Node(
            package='ros_gz_image',
            executable='image_bridge',
            name=f'video_bridge_{i}',
            arguments=[f'/world/default/model/x500_depth_{i}/link/camera_link/sensor/thermal_camera/image'],
            output='screen'
        )

        # SPARRING PATCH: Stagger each drone's ignition by 15 seconds
        delay = float(i * 15)
        delayed_launch = TimerAction(
            period=delay,
            actions=[commander, vision, bridge]
        )
        ld.add_action(delayed_launch)

    return ld
