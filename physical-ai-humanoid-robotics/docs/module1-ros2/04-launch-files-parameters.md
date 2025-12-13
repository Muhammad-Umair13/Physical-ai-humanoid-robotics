<!-- ---
sidebar_label: 'Launch Files and Parameters'
sidebar_position: 4
---

# Launch Files and Parameters

## Overview

In complex robotic systems like humanoid robots, managing multiple nodes and their configurations can become challenging. ROS 2 provides launch files and parameters to simplify the process of starting and configuring multiple nodes simultaneously.

## Launch Files

Launch files allow you to start multiple nodes with a single command, configure their parameters, and manage their lifecycle. They can be written in Python, XML, or YAML.

### Python Launch Files

Python launch files provide the most flexibility and are the recommended approach for complex systems:

```python
# example_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # Launch a node
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker_node',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'frequency': 1.0}
            ],
            remappings=[
                ('chatter', 'my_chatter')
            ]
        ),

        # Launch another node
        Node(
            package='demo_nodes_cpp',
            executable='listener',
            name='listener_node',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        )
    ])
```

### Launch File Concepts

#### Actions
- `Node`: Launch a ROS node
- `DeclareLaunchArgument`: Define a launch argument
- `LogInfo`: Print a message to the console
- `TimerAction`: Execute actions after a delay
- `ExecuteProcess`: Run an external process

#### Substitutions
- `LaunchConfiguration`: Access launch argument values
- `PathJoinSubstitution`: Join path components
- `TextSubstitution`: Literal text
- `PythonExpression`: Python expressions

### XML Launch Files

XML launch files provide a more declarative approach:

```xml
<launch>
  <arg name="use_sim_time" default="false"/>

  <node pkg="demo_nodes_cpp" exec="talker" name="talker_node">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
    <param name="frequency" value="1.0"/>
    <remap from="chatter" to="my_chatter"/>
  </node>

  <node pkg="demo_nodes_cpp" exec="listener" name="listener_node">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>
</launch>
```

### YAML Launch Files

YAML launch files offer a more readable format:

```yaml
launch:
  - node:
      pkg: "demo_nodes_cpp"
      exec: "talker"
      name: "talker_node"
      parameters:
        - use_sim_time: $(var use_sim_time)
        - frequency: 1.0
      remappings:
        - ["chatter", "my_chatter"]
```

## Parameters

Parameters in ROS 2 allow you to configure nodes at runtime. They can be declared, set, and changed dynamically.

### Parameter Declaration

```python
# Python parameter declaration
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_enabled', True)

        # Access parameter values
        robot_name = self.get_parameter('robot_name').value
        max_velocity = self.get_parameter('max_velocity').value
        safety_enabled = self.get_parameter('safety_enabled').value
```

```cpp
// C++ parameter declaration
class ParameterNode : public rclcpp::Node
{
public:
  ParameterNode() : Node("parameter_node")
  {
    // Declare parameters
    this->declare_parameter("robot_name", "humanoid_robot");
    this->declare_parameter("max_velocity", 1.0);
    this->declare_parameter("safety_enabled", true);

    // Get parameter values
    std::string robot_name = this->get_parameter("robot_name").as_string();
    double max_velocity = this->get_parameter("max_velocity").as_double();
    bool safety_enabled = this->get_parameter("safety_enabled").as_bool();
  }
};
```

### Parameter Callbacks

You can react to parameter changes:

```python
from rcl_interfaces.msg import ParameterEvent

def parameter_callback(self, parameter_list):
    for param in parameter_list.parameters:
        if param.name == 'max_velocity':
            new_value = param.value.double_value
            self.get_logger().info(f'Max velocity changed to: {new_value}')
            # Handle the parameter change
            self.update_velocity_limits(new_value)
    return SetParametersResult(successful=True)

# Register the callback
self.add_on_set_parameters_callback(self.parameter_callback)
```

### Parameter Files (YAML)

Parameters can be loaded from YAML files:

```yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false

parameter_node:  # Applies to node named 'parameter_node'
  ros__parameters:
    robot_name: "advanced_humanoid"
    max_velocity: 2.0
    safety_enabled: true
    joint_limits:
      left_leg:
        min: -1.57
        max: 1.57
      right_arm:
        min: -2.0
        max: 2.0
```

## Advanced Launch Features

### Conditional Launch

Launch nodes based on conditions:

```python
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

use_gui = LaunchConfiguration('use_gui')

# Launch node only if use_gui is true
Node(
    package='rviz2',
    executable='rviz2',
    condition=IfCondition(use_gui)
),

# Launch node unless use_gui is true
Node(
    package='rqt_graph',
    executable='rqt_graph',
    condition=UnlessCondition(use_gui)
)
```

### Composition

Run multiple nodes in the same process for better performance:

```python
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

container = ComposableNodeContainer(
    name='image_processing_container',
    namespace='',
    package='rclcpp_components',
    executable='component_container',
    composable_node_descriptions=[
        ComposableNode(
            package='image_proc',
            plugin='image_proc::RectifyNode',
            name='rectify_node'
        ),
        ComposableNode(
            package='image_view',
            plugin='image_view::ImageViewNode',
            name='image_view_node'
        )
    ]
)
```

### Lifecycle Nodes

Manage node lifecycle explicitly:

```python
from launch_ros.actions import LifecycleNode

LifecycleNode(
    package='my_package',
    executable='my_lifecycle_node',
    name='my_lifecycle_node',
    namespace='',
    parameters=[...]
)
```

## Practical Applications in Humanoid Robotics

### Robot Bringup

A typical humanoid robot launch file might include:

```python
def generate_launch_description():
    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[robot_description]
        ),

        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        # IMU driver
        Node(
            package='imu_driver',
            executable='imu_node',
            parameters=[
                {'sensor_port': '/dev/ttyUSB0'},
                {'baud_rate': 115200}
            ]
        ),

        # Controller manager
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[controller_config]
        )
    ])
```

### Simulation vs Real Robot

Different configurations for simulation and real hardware:

```python
# Simulation-specific launch
sim_nodes = [
    Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'humanoid_robot']
    )
]

# Real robot-specific launch
real_nodes = [
    Node(
        package='real_robot_driver',
        executable='hardware_interface'
    )
]

# Conditionally include based on argument
DeclareLaunchArgument('sim_mode', default_value='false'),
# ...
# Use IfCondition(sim_mode) to choose between sim_nodes and real_nodes
```

### Parameter Management for Different Robot Configurations

Different humanoid robots may have different joint configurations:

```yaml
# For Atlas robot
atlas_config:
  ros__parameters:
    joints:
      - head_pan
      - head_tilt
      - left_arm_shoulder_pitch
      - left_arm_shoulder_roll
      # ... more joints

# For NAO robot
nao_config:
  ros__parameters:
    joints:
      - HeadYaw
      - HeadPitch
      - LShoulderPitch
      - LShoulderRoll
      # ... more joints
```

## Best Practices

### Launch File Organization
- Use descriptive names for launch files
- Group related functionality in separate launch files
- Use include directives to compose complex launch systems
- Document launch arguments and their purposes

### Parameter Management
- Group related parameters logically
- Use meaningful parameter names
- Provide appropriate default values
- Document parameter meanings and valid ranges
- Use parameter files for complex configurations

### Performance Considerations
- Use composition for nodes that communicate frequently
- Consider process vs thread boundaries for performance
- Use appropriate QoS settings in launch files
- Monitor resource usage during launch

## Summary

Launch files and parameters are essential tools for managing complex robotic systems. They allow you to configure and start multiple nodes with a single command, making it easier to manage humanoid robotics applications. Proper use of these tools leads to more maintainable and configurable robotic systems. -->