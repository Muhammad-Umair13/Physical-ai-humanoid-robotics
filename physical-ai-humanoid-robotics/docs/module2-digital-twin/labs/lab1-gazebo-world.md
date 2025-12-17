---
sidebar_label: 'Lab 1: Creating a Gazebo World for Humanoid Robot Testing'
sidebar_position: 1
---

# Lab 1: Creating a Gazebo World for Humanoid Robot Testing

## Objective

In this lab, you will create a custom Gazebo world specifically designed for testing humanoid robot behaviors. You will learn to design environments with various obstacles, terrains, and challenges that humanoid robots might encounter in real-world scenarios.

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Gazebo Garden (or compatible version) installed
- Basic understanding of URDF/XACRO robot modeling
- Basic knowledge of ROS 2 concepts (topics, nodes)

## Learning Outcomes

By the end of this lab, you will be able to:
1. Create custom Gazebo world files using SDF format
2. Design environments with different terrains and obstacles
3. Configure physics parameters for realistic simulation
4. Spawn and test robots in custom environments
5. Evaluate robot performance in simulated scenarios

## Step 1: Setting Up the Workspace

First, create a new ROS 2 package for our simulation:

```bash
# Create a new workspace if you don't have one
mkdir -p ~/gazebo_ws/src
cd ~/gazebo_ws/src

# Create the simulation package
ros2 pkg create --build-type ament_cmake gazebo_humanoid_simulation
cd gazebo_humanoid_simulation
```

Create the necessary directory structure:

```bash
mkdir -p worlds models launch
```

## Step 2: Creating the Basic World File

Create a new world file at `gazebo_humanoid_simulation/worlds/humanoid_test_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <!-- Include default ground plane and lighting -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Indoor testing area with walls -->
    <model name="testing_room_walls">
      <static>true</static>

      <!-- North wall -->
      <link name="north_wall">
        <pose>0 5 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>

      <!-- South wall -->
      <link name="south_wall">
        <pose>0 -5 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>

      <!-- East wall -->
      <link name="east_wall">
        <pose>5 0 1.5 0 0 1.5707</pose>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>

      <!-- West wall -->
      <link name="west_wall">
        <pose>-5 0 1.5 0 0 1.5707</pose>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacles for humanoid navigation -->
    <model name="obstacle_1">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.5 0.5 1.0</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.5 0.5 1.0</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>1.0 0.3 0.3 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.083</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.083</iyy>
            <iyz>0</iyz>
            <izz>0.083</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Stairs for humanoid climbing test -->
    <model name="stairs">
      <static>true</static>
      <link name="step_1">
        <pose>-2 -3 0.1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="step_2">
        <pose>-2 -3 0.3 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="step_3">
        <pose>-2 -3 0.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Ramp for incline testing -->
    <model name="ramp">
      <static>true</static>
      <link name="ramp_link">
        <pose>3 0 0 0 0.2 0</pose>
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>file://meshes/ramp.dae</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>file://meshes/ramp.dae</uri>
            </mesh>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Lighting -->
    <light name="room_light" type="point">
      <pose>0 0 3 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>20</range>
        <constant>0.2</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
    </light>
  </world>
</sdf>
```

## Step 3: Creating a Simple Humanoid Robot Model

Create a simple humanoid model for testing at `gazebo_humanoid_simulation/models/simple_humanoid/model.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_humanoid">
    <!-- Torso -->
    <link name="torso">
      <pose>0 0 1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box><size>0.3 0.3 0.6</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.3 0.3 0.6</size></box>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.8 1</ambient>
          <diffuse>0.2 0.2 1.0 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.5</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.5</iyy>
          <iyz>0</iyz>
          <izz>0.3</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Head -->
    <link name="head">
      <pose>0 0 1.4 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <sphere><radius>0.15</radius></sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere><radius>0.15</radius></sphere>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>1.0 1.0 1.0 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.02</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.02</iyy>
          <iyz>0</iyz>
          <izz>0.02</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Left leg -->
    <link name="left_thigh">
      <pose>-0.1 0 0.7 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.8 0.2 0.2 1</ambient>
          <diffuse>1.0 0.3 0.3 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>3.0</mass>
        <inertia>
          <ixx>0.04</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.04</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
    </link>

    <link name="left_shin">
      <pose>-0.1 0 0.3 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.8 0.2 0.2 1</ambient>
          <diffuse>1.0 0.3 0.3 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>2.5</mass>
        <inertia>
          <ixx>0.03</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.03</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
    </link>

    <link name="left_foot">
      <pose>-0.1 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box><size>0.2 0.1 0.08</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.2 0.1 0.08</size></box>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.4 0.4 0.4 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.002</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.005</iyy>
          <iyz>0</iyz>
          <izz>0.005</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Right leg (similar to left, mirrored) -->
    <link name="right_thigh">
      <pose>0.1 0 0.7 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.08</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.8 0.2 0.2 1</ambient>
          <diffuse>1.0 0.3 0.3 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>3.0</mass>
        <inertia>
          <ixx>0.04</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.04</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
    </link>

    <link name="right_shin">
      <pose>0.1 0 0.3 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.07</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.8 0.2 0.2 1</ambient>
          <diffuse>1.0 0.3 0.3 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>2.5</mass>
        <inertia>
          <ixx>0.03</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.03</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
    </link>

    <link name="right_foot">
      <pose>0.1 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box><size>0.2 0.1 0.08</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.2 0.1 0.08</size></box>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.4 0.4 0.4 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.002</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.005</iyy>
          <iyz>0</iyz>
          <izz>0.005</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Joints -->
    <!-- Torso to head -->
    <joint name="neck_joint" type="revolute">
      <parent>torso</parent>
      <child>head</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>100</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Left leg joints -->
    <joint name="left_hip_joint" type="revolute">
      <parent>torso</parent>
      <child>left_thigh</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>200</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="left_knee_joint" type="revolute">
      <parent>left_thigh</parent>
      <child>left_shin</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>0</lower>
          <upper>2.35</upper>
          <effort>200</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="left_ankle_joint" type="revolute">
      <parent>left_shin</parent>
      <child>left_foot</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>100</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Right leg joints -->
    <joint name="right_hip_joint" type="revolute">
      <parent>torso</parent>
      <child>right_thigh</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>200</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="right_knee_joint" type="revolute">
      <parent>right_thigh</parent>
      <child>right_shin</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>0</lower>
          <upper>2.35</upper>
          <effort>200</effort>
          <velocity>2</velocity>
        </limit>
      </axis>
    </joint>

    <joint name="right_ankle_joint" type="revolute">
      <parent>right_shin</parent>
      <child>right_foot</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>100</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

Create the model configuration file at `gazebo_humanoid_simulation/models/simple_humanoid/model.config`:

```xml
<?xml version="1.0"?>
<model>
  <name>simple_humanoid</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>
    A simple humanoid robot model for testing in Gazebo.
  </description>
</model>
```

## Step 4: Creating a Launch File

Create a launch file at `gazebo_humanoid_simulation/launch/humanoid_world.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='humanoid_test_world.sdf',
        description='Choose one of the world files from `/gazebo_humanoid_simulation/worlds`'
    )

    # Launch Gazebo with our world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('gazebo_humanoid_simulation'),
                'worlds',
                LaunchConfiguration('world')
            ])
        }.items()
    )

    # Spawn the humanoid robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        spawn_entity
    ])
```

## Step 5: Building and Running the Simulation

Build the package:

```bash
cd ~/gazebo_ws
colcon build --packages-select gazebo_humanoid_simulation
source install/setup.bash
```

Run the simulation:

```bash
# Set the GAZEBO_MODEL_PATH to include our models
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/gazebo_ws/src/gazebo_humanoid_simulation/models

# Launch the simulation
ros2 launch gazebo_humanoid_simulation humanoid_world.launch.py
```

## Step 6: Testing Robot Behaviors

Once the simulation is running, you can test various humanoid behaviors:

### Balance Testing
1. Observe the humanoid's initial balance when spawned
2. Apply external forces to test stability
3. Check if the robot maintains balance on different surfaces

### Navigation Testing
1. Test movement around obstacles
2. Try climbing the stairs
3. Navigate up the ramp

### Interaction Testing
1. Check how the robot interacts with static objects
2. Test collision detection and response
3. Observe physics behavior when contacting surfaces

## Step 7: Experimenting with Physics Parameters

Try modifying physics parameters in your world file to see different effects:

- Change gravity values to simulate different environments
- Adjust friction coefficients for different surface properties
- Modify contact parameters to change collision behavior
- Experiment with different solver parameters for performance vs. accuracy

## Expected Results

After completing this lab, you should have:
- A custom Gazebo world with various challenges for humanoid robots
- A simple humanoid robot model that can be spawned in the world
- A working launch file to start the simulation
- Understanding of how to create and customize Gazebo environments

## Troubleshooting

If you encounter issues:

1. **Robot falls through the ground**: Check that the ground plane is properly included and that the robot has proper collision geometry
2. **Physics instability**: Adjust solver parameters in the world file
3. **Models not loading**: Verify GAZEBO_MODEL_PATH is set correctly
4. **Performance issues**: Simplify collision geometry or adjust physics parameters

## Extensions

Try these extensions to deepen your understanding:

1. Add more complex obstacles (narrow passages, moving platforms)
2. Create outdoor environments with terrain
3. Add sensor models (cameras, IMUs, lidars) to the humanoid
4. Implement simple control algorithms to make the robot move
5. Add lighting effects and environmental conditions

## Summary

In this lab, you've created a comprehensive Gazebo world specifically designed for testing humanoid robot behaviors. You've learned how to structure world files, create simple robot models, and launch simulations with custom environments. This foundation will help you develop more complex simulation scenarios for testing humanoid robotics applications.