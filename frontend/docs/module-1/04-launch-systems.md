---
sidebar_position: 4
---

# ROS 2 Launch Systems

Modern robotic systems consist of many interconnected nodes, each handling a specific function. Starting these nodes individually is impractical—launch systems coordinate the initialization, configuration, and lifecycle of entire robotic applications. This chapter covers ROS 2's launch system, from basic launch files through advanced composition patterns.

## Understanding ROS 2 Launch Architecture

ROS 2 introduced a new Python-based launch system that replaced the XML-based approach of ROS 1. This new system offers greater flexibility, better integration with ROS 2 concepts, and improved support for complex deployment scenarios.

### Launch System Components

The ROS 2 launch system consists of several layers:

**Launch Service**: The core runtime that executes launch files and manages the lifecycle of launched processes.

**Launch Descriptions**: Declarative representations of what to launch, expressed as Python objects.

**Launch Actions**: Individual units of work—starting nodes, setting parameters, executing shell commands.

**Launch Substitutions**: Late-binding values resolved at launch time, enabling conditional and parameterized launches.

### Basic Launch File Structure

A minimal launch file demonstrates the fundamental concepts:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Generate the launch description for the robot system."""

    # Create a node action - this will start a ROS 2 node
    chatter_node = Node(
        package='demo_nodes_cpp',
        executable='talker',
        name='my_talker',  # Override the default node name
        output='screen',   # Print output to console
        parameters=[       # Configuration parameters
            {'publish_rate': 10.0},
            {'message_prefix': 'Hello from launch'}
        ]
    )

    # Another node
    listener_node = Node(
        package='demo_nodes_py',
        executable='listener',
        name='my_listener',
        output='screen'
    )

    # The launch description contains all launch actions
    return LaunchDescription([
        chatter_node,
        listener_node,
    ])
```

Save this as `demo.launch.py` and run:

```bash
ros2 launch package_name demo.launch.py
```

### LaunchDescription and Actions

`LaunchDescription` is the container for all launch actions. Common action types include:

**Node Action**: Starts a ROS 2 node:

```python
from launch_ros.actions import Node

camera_node = Node(
    package='camera_driver',
    executable='camera_node',
    name='front_camera',
    namespace='sensors',
    output='screen',
    respawn=True,           # Restart if process dies
    respawn_delay=2.0,      # Wait before restart
    parameters=[
        {'camera_id': 0},
        {'resolution': [1920, 1080]},
        {'frame_rate': 30.0}
    ],
    remappings=[
        ('image_raw', 'camera/image'),
        ('camera_info', 'camera/camera_info')
    ],
    environment={
        'CUDA_VISIBLE_DEVICES': '0'
    }
)
```

**ExecuteProcess Action**: Runs arbitrary commands:

```python
from launch.actions import ExecuteProcess

gazebo_cmd = ExecuteProcess(
    cmd=['gzserver', '--physics', 'ode', '-s'],
    output='screen',
    shell=True  # Run through shell for pipe/redirection
)

# Execute a setup script before starting nodes
setup_script = ExecuteProcess(
    cmd=['source', 'install/setup.bash'],
    shell=True
)
```

**IncludeLaunchDescription**: Includes other launch files:

```python
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindPackageShare

# Include another launch file
navigation_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([
        FindPackageShare('nav2_bringup'),
        '/launch/navigation_launch.py'
    ]),
    launch_arguments={
        'params_file': '/path/to/params.yaml',
        'use_sim_time': 'true'
    }.items()
)
```

**SetEnvironmentVariable**: Set environment variables:

```python
from launch.actions import SetEnvironmentVariable

env_setup = SetEnvironmentVariable(
    name='AMENT_PREFIX_PATH',
    value='/opt/ros/humble'
)
```

## Parameter Management in Launch Files

Launch files provide powerful parameter management capabilities.

### Loading Parameters from YAML

```python
from launch_ros.actions import Node
from launch.substitutions import FindPackageShare
from launch_ros.substitutions import ParameterFile

# Load all parameters from a YAML file
robot_node = Node(
    package='robot_control',
    executable='robot_controller',
    parameters=[
        ParameterFile(
            param_file=FindPackageShare('robot_config') / 'config/params.yaml',
            allow_substs=True
        )
    ]
)
```

YAML parameter files:

```yaml
# params.yaml
robot_controller:
    ros__parameters:
        # Simple values
        publish_rate: 50.0
        use_sim_time: false

        # Nested structure
        joint_limits:
            hip_joint:
                min_position: -1.57
                max_position: 1.57
                max_velocity: 5.0
            knee_joint:
                min_position: -2.5
                max_position: 0.0

        # Arrays
        enabled_joints: ['left_hip', 'right_hip', 'left_knee', 'right_knee']

        # Booleans and strings
        enable_logging: true
        log_level: 'INFO'
```

### Parameter Overrides

Launch arguments can override YAML parameters:

```python
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

# Declare launch arguments
publish_rate_arg = DeclareLaunchArgument(
    'publish_rate',
    default_value='50.0',
    description='Rate for publishing robot state (Hz)'
)

use_sim_time_arg = DeclareLaunchArgument(
    'use_sim_time',
    default_value='false',
    choices=['true', 'false'],
    description='Use simulation time'
)

# Use arguments in node configuration
robot_node = Node(
    package='robot_control',
    executable='robot_controller',
    parameters=[
        {'publish_rate': LaunchConfiguration('publish_rate')},
        {'use_sim_time': LaunchConfiguration('use_sim_time')}
    ]
)
```

## Conditional Launching

Launch files can conditionally include actions based on arguments:

```python
from launch.actions import DeclareLaunchArgument
from launch.actions import LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

# Declare argument for selection
robot_type_arg = DeclareLaunchArgument(
    'robot_type',
    default_value='atlas',
    description='Type of robot: atlas, hubo, custom',
    choices=['atlas', 'hubo', 'custom']
)

# Conditional action - only run for specific robot type
atlas_specific = Node(
    package='atlas_control',
    executable='atlas_controller',
    condition=IfCondition(
        PythonExpression(["'", LaunchConfiguration('robot_type'), "' == 'atlas'"])
    )
)

# Run only when NOT simulation
real_robot_only = LogInfo(
    msg='Running with real robot hardware',
    condition=UnlessCondition(LaunchConfiguration('simulation'))
)

# Disable hardware for simulation
simulation_mode = LogInfo(
    msg='Running in simulation mode',
    condition=IfCondition(LaunchConfiguration('simulation'))
)
```

## Environment-Specific Configurations

Launch files commonly handle different environments:

```python
def generate_launch_description():
    # Arguments for environment selection
    simulation_arg = DeclareLaunchArgument(
        'simulation',
        default_value='false',
        description='True for Gazebo simulation'
    )

    robot_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.100',
        description='IP address of robot hardware'
    )

    # Common nodes for all environments
    state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': get_robot_description(),
            'use_sim_time': LaunchConfiguration('simulation')
        }]
    )

    # Simulation-specific nodes
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ]),
        condition=IfCondition(LaunchConfiguration('simulation')),
        launch_arguments={
            'world': FindPackageShare('robot_gazebo') / 'worlds/playground.world'
        }.items()
    )

    # Hardware-specific nodes
    hardware_bridge = Node(
        package='hardware_bridge',
        executable='ethercat_bridge',
        condition=UnlessCondition(LaunchConfiguration('simulation')),
        parameters=[{
            'robot_ip': LaunchConfiguration('robot_ip')
        }]
    )

    return LaunchDescription([
        simulation_arg,
        robot_arg,
        state_publisher,
        gazebo,
        hardware_bridge,
    ])
```

## Composable Nodes and Component Containers

ROS 2's composition system allows nodes to be dynamically loaded into a single process:

```python
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    """Launch a perception pipeline with composable nodes."""

    # Container that holds multiple components
    perception_container = ComposableNodeContainer(
        name='perception_container',
        namespace='perception',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # Camera driver as a component
            ComposableNode(
                package='camera_driver',
                plugin='camera_driver::CameraDriver',
                name='camera_driver',
                parameters=[{
                    'camera_id': 0,
                    'image_width': 1280,
                    'image_height': 720
                }]
            ),

            # Object detection component
            ComposableNode(
                package='object_detection',
                plugin='object_detection::YoloDetector',
                name='object_detector',
                parameters=[{
                    'model_path': '/models/yolov8.onnx',
                    'confidence_threshold': 0.5
                }],
                remappings=[
                    ('image_in', '/camera/image_raw'),
                    ('detections', '/perception/detections')
                ]
            ),

            # Tracking component
            ComposableNode(
                package='object_tracking',
                plugin='tracking::ByteTrack',
                name='tracker',
                parameters=[{
                    'max_age': 30,
                    'min_hits': 3
                }]
            )
        ],
        output='screen'
    )

    return LaunchDescription([perception_container])
```

### Advantages of Composition

1. **Lower Latency**: Intra-process communication avoids serialization
2. **Reduced Memory**: Single process vs multiple processes
3. **Simpler Debugging**: Single process to attach debugger to
4. **Resource Sharing**: Shared memory for large data

## Event Handling and Callbacks

Launch systems can respond to events during execution:

```python
from launch import LaunchDescription, LaunchContext
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.events import Shutdown

def generate_launch_description():
    """Launch with event handling."""

    # Event handler for process exit
    on_controller_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=controller_node,
            on_exit=[
                LogInfo(msg='Controller exited, restarting...'),
                ExecuteProcess(
                    cmd=['systemctl', 'restart', 'robot-controller'],
                    shell=True
                )
            ]
        )
    )

    # Event handler for stdout/stderr
    on_controller_output = RegisterEventHandler(
        OnProcessIO(
            target_action=controller_node,
            on_stdout=lambda context: print(context.locals['text']),
            on_stderr=lambda context: print(f'ERROR: {context.locals["text"]}')
        )
    )

    # Shutdown handler
    on_emergency_stop = RegisterEventHandler(
        OnProcessExit(
            target_action=estop_node,
            on_exit=[
                EmitEvent(event=Shutdown(
                    reason='Emergency stop pressed'
                ))
            ]
        )
    )

    return LaunchDescription([
        controller_node,
        estop_node,
        on_controller_exit,
        on_controller_output,
        on_emergency_stop,
    ])
```

## Best Practices

### Modular Launch Files

Structure launch files for reusability:

```
launch/
├── robot.launch.py          # Main entry point
├── description.launch.py    # Robot description
├── control.launch.py        # Control nodes
├── perception.launch.py     # Perception pipeline
└── navigation.launch.py     # Navigation stack
```

```python
# robot.launch.py - Main entry point
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('robot_description'),
                '/launch/description.launch.py'
            ])
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('robot_control'),
                '/launch/control.launch.py'
            ])
        ),
    ])
```

### Parameter Organization

Group parameters logically and use separate files:

```
config/
├── control.yaml            # Controller parameters
├── perception.yaml         # Camera, detection params
├── navigation.yaml         # Navigation parameters
└── hardware.yaml           # Hardware-specific settings
```

### Logging and Debugging

Configure output appropriately:

```python
# For debugging: show all output
debug_node = Node(
    package='debug_pkg',
    executable='debug_node',
    output='screen',
    prefix='xterm -e'  # Open in new terminal
)

# For production: log to file
production_node = Node(
    package='production_pkg',
    executable='production_node',
    output='log',
    log_output='file',
    suffix=['2>&1', '|', 'tee', '-a', '/var/log/robot.log']
)
```

### Testing Launch Files

```python
import unittest
from launch import LaunchDescription
from launch_testing import generate_test_description
from launch_testing.asserts import assertSequentialStdout

class TestLaunchFiles(unittest.TestCase):

    def test_launch_file_generates(self):
        """Test that launch file generates without errors."""
        ld = generate_launch_description()
        self.assertIsInstance(ld, LaunchDescription)

    def test_required_nodes(self):
        """Test that required nodes are included."""
        ld = generate_launch_description()
        node_names = [action.name for action in ld.entities
                      if hasattr(action, 'name')]

        self.assertIn('robot_state_publisher', node_names)
        self.assertIn('joint_state_publisher', node_names)
```

## Key Takeaways

Launch systems are essential for managing complex ROS 2 applications:

- **Python-based launch** offers flexibility and power
- **Launch descriptions** organize actions into structured deployments
- **Parameters** enable configuration without code changes
- **Conditionals** support environment-specific behavior
- **Composition** improves performance through intra-process communication
- **Modular design** enables reusability across projects

With launch systems mastered, we now turn to simulation—creating digital twins of robotic systems.
