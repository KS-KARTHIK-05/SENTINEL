"""Launch the SENTINEL ROS 2 stack against a running SITL (scripts/launch_sitl.sh).

    ros2 launch sentinel_ai swarm_launch.py n_uav:=3 world:=sentinel_d25_s0 \
        method:=sentinel model_path:=/path/to/model.pt failures:='[{"t":100,"uav":1,"kind":"crash"}]'

Per UAV i (namespace /px4_i): ros_gz image bridge, thermal_vision, flight_commander,
emps_agent. Global: clock bridge, comm_emulator, chaos_monkey, metrics_logger.
Spawn poses are read from $SENTINEL_RUN_DIR/spawn.csv written by launch_sitl.sh.
"""
import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _world_dir():
    try:
        return os.path.join(get_package_share_directory('sentinel_gazebo'), 'worlds')
    except PackageNotFoundError:
        here = os.path.dirname(os.path.realpath(__file__))
        return os.path.normpath(os.path.join(here, '..', '..', 'sentinel_gazebo', 'worlds'))


def _setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in
         ('n_uav', 'world', 'method', 'channel_profile', 'model_path', 'failures', 'seed',
          'run_dir', 'out_dir', 'altitude', 'world_dir', 'use_sim_time', 'perception',
          't_hold_s', 'tuned_yaml')}
    n = int(a['n_uav'])
    world_dir = a['world_dir'] or _world_dir()
    truth = os.path.join(world_dir, a['world'] + '.json')
    with open(os.path.join(os.path.expanduser(a['run_dir']), 'spawn.csv')) as f:
        spawn = [tuple(map(float, line.split(','))) for line in f if line.strip()]
    sim_time = a['use_sim_time'].lower() == 'true'
    common = {'use_sim_time': sim_time}
    shared = {**common, 'n_uav': n, 'truth_json': truth, 'channel_profile': a['channel_profile'],
              'seed': int(a['seed']), 'method': a['method'], 'tuned_yaml': a['tuned_yaml']}
    nodes = [
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='clock_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'], output='log'),
        Node(package='sentinel_ai', executable='comm_emulator', parameters=[shared], output='screen'),
        Node(package='sentinel_ai', executable='chaos_monkey',
             parameters=[{**common, 'schedule': a['failures']}], output='screen'),
        Node(package='sentinel_ai', executable='metrics_logger',
             parameters=[{**shared, 'altitude': float(a['altitude']),
                          'out_dir': os.path.expanduser(a['out_dir'])}], output='screen'),
    ]
    for i in range(n):
        ns = f'px4_{i}'
        img = (f'/world/{a["world"]}/model/x500_thermal_{i}/link/thermal_link'
               '/sensor/thermal_camera/image')
        per = {**common, 'uav_id': i, 'spawn_x': spawn[i][0], 'spawn_y': spawn[i][1],
               'altitude': float(a['altitude'])}
        nodes.append(Node(package='sentinel_ai', executable='flight_commander', namespace=ns,
                          parameters=[per], output='screen'))
        if a['perception'] == 'yolo':
            # deployment path: Gazebo thermal camera -> ROS -> onboard YOLO detector
            nodes += [
                Node(package='ros_gz_image', executable='image_bridge',
                     name=f'thermal_bridge_{i}', arguments=[img], output='log'),
                Node(package='sentinel_ai', executable='thermal_vision', namespace=ns,
                     parameters=[{**per, 'image_topic': img, 'model_path': a['model_path'],
                                  'publish_debug': True}],
                     output='screen'),
            ]
        else:
            # E8 cross-validation: same calibrated detector model as the fast simulator
            nodes.append(Node(package='sentinel_ai', executable='sim_detector', namespace=ns,
                              parameters=[{**per, 'truth_json': truth, 'seed': int(a['seed'])}],
                              output='screen'))
        agent_params = {**per, **shared}
        if a['t_hold_s']:
            agent_params['t_hold_s'] = float(a['t_hold_s'])
        nodes.append(Node(package='sentinel_ai', executable='emps_agent', namespace=ns,
                          parameters=[agent_params], output='screen'))
    return nodes


def generate_launch_description():
    args = [
        DeclareLaunchArgument('n_uav', default_value='3'),
        DeclareLaunchArgument('world', default_value='sentinel_d25_s0'),
        DeclareLaunchArgument('world_dir', default_value=''),
        DeclareLaunchArgument('method', default_value='sentinel'),
        DeclareLaunchArgument('channel_profile', default_value='urban_rubble'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('failures', default_value='[]'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('altitude', default_value='30.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('perception', default_value='yolo'),     # yolo | model
        DeclareLaunchArgument('t_hold_s', default_value=''),
        DeclareLaunchArgument('tuned_yaml', default_value=''),
        DeclareLaunchArgument('run_dir',
                              default_value=os.environ.get('SENTINEL_RUN_DIR', '~/.ros/sentinel_sitl')),
        DeclareLaunchArgument('out_dir', default_value='~/.ros/sentinel_sitl/metrics'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
