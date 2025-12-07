---
sidebar_label: 'Lab 1: ROS 2 Node Sensor Integration'
sidebar_position: 1
---

# Lab 1: ROS 2 Node Sensor Integration

## Objective

In this lab, you will create a ROS 2 node that simulates sensor data from a humanoid robot and another node that processes this data. This will help you understand the publish/subscribe communication pattern in ROS 2 and how to handle sensor data in a robotic system.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Basic knowledge of Python or C++
- Docusaurus running for the textbook (if testing the RAG chatbot integration)

## Learning Outcomes

By the end of this lab, you will be able to:
1. Create a publisher node that simulates sensor data
2. Create a subscriber node that processes sensor data
3. Configure Quality of Service (QoS) settings appropriately for sensor data
4. Use launch files to start multiple nodes simultaneously
5. Test your nodes using ROS 2 command-line tools

## Step 1: Create the Sensor Publisher Node

Create a new ROS 2 package for this lab:

```bash
# Create a new workspace if you don't have one
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python sensor_integration_lab
cd sensor_integration_lab
```

Create the sensor publisher node in `sensor_integration_lab/sensor_publisher.py`:

```python
#!/usr/bin/env python3

"""
Sensor Publisher Node for Humanoid Robot
Publishes simulated sensor data including IMU, joint states, and camera data
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Header
import math
import random


class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publishers for different sensor types
        self.imu_publisher = self.create_publisher(Imu, 'imu/data', 10)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create a timer to publish data at regular intervals
        self.timer = self.create_timer(0.1, self.publish_sensor_data)  # 10 Hz

        # Initialize joint names for a simple humanoid model
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        self.get_logger().info('Sensor Publisher Node has started')

    def publish_sensor_data(self):
        """Publish simulated sensor data"""
        # Publish IMU data
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Simulate IMU readings (acceleration, angular velocity, orientation)
        imu_msg.linear_acceleration.x = random.uniform(-0.1, 0.1)
        imu_msg.linear_acceleration.y = random.uniform(-0.1, 0.1)
        imu_msg.linear_acceleration.z = 9.8 + random.uniform(-0.2, 0.2)

        imu_msg.angular_velocity.x = random.uniform(-0.05, 0.05)
        imu_msg.angular_velocity.y = random.uniform(-0.05, 0.05)
        imu_msg.angular_velocity.z = random.uniform(-0.05, 0.05)

        # Simple orientation (keeping it upright for now)
        imu_msg.orientation.w = 1.0
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0

        self.imu_publisher.publish(imu_msg)

        # Publish Joint State data
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names

        # Simulate joint positions with small random movements
        joint_msg.position = []
        for i, name in enumerate(self.joint_names):
            # Create different movement patterns for different joint types
            if 'hip' in name:
                # Hip joints - simulate walking pattern
                position = math.sin(self.get_clock().now().nanoseconds / 1e9) * 0.2
            elif 'knee' in name:
                # Knee joints - follow hip movement
                position = math.sin(self.get_clock().now().nanoseconds / 1e9) * 0.1
            elif 'ankle' in name:
                # Ankle joints - smaller movement
                position = math.sin(self.get_clock().now().nanoseconds / 1e9 + 0.5) * 0.05
            elif 'shoulder' in name:
                # Shoulder joints - arm movement
                position = math.sin(self.get_clock().now().nanoseconds / 1e9 + 1.0) * 0.3
            elif 'elbow' in name:
                # Elbow joints - follow shoulder
                position = math.sin(self.get_clock().now().nanoseconds / 1e9 + 1.5) * 0.2
            else:
                position = 0.0

            joint_msg.position.append(position)

        # Set velocities and efforts to zero for simulation
        joint_msg.velocity = [0.0] * len(self.joint_names)
        joint_msg.effort = [0.0] * len(self.joint_names)

        self.joint_state_publisher.publish(joint_msg)

        self.get_logger().debug(f'Published sensor data at {imu_msg.header.stamp.sec}.{imu_msg.header.stamp.nanosec}')


def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 2: Create the Sensor Processing Node

Create the sensor processing node in `sensor_integration_lab/sensor_processor.py`:

```python
#!/usr/bin/env python3

"""
Sensor Processor Node for Humanoid Robot
Subscribes to sensor data and processes it for control decisions
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64
import math


class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Create subscribers for sensor data
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Create publishers for processed data
        self.balance_publisher = self.create_publisher(Float64, 'balance_state', 10)
        self.stability_publisher = self.create_publisher(Float64, 'stability_index', 10)

        # Initialize variables to store sensor data
        self.imu_data = None
        self.joint_data = None

        self.get_logger().info('Sensor Processor Node has started')

    def imu_callback(self, msg):
        """Process IMU data to determine balance state"""
        self.imu_data = msg

        # Calculate balance based on IMU readings
        # This is a simplified balance calculation
        linear_accel = math.sqrt(
            msg.linear_acceleration.x**2 +
            msg.linear_acceleration.y**2 +
            msg.linear_acceleration.z**2
        )

        # Calculate angular velocity magnitude
        angular_vel = math.sqrt(
            msg.angular_velocity.x**2 +
            msg.angular_velocity.y**2 +
            msg.angular_velocity.z**2
        )

        # Balance state: 0 = perfectly balanced, 1 = unstable
        balance_state = min(1.0, (abs(msg.linear_acceleration.y) + angular_vel) / 2.0)

        # Publish balance state
        balance_msg = Float64()
        balance_msg.data = balance_state
        self.balance_publisher.publish(balance_msg)

        # Calculate stability index
        stability_index = 1.0 - balance_state
        stability_msg = Float64()
        stability_msg.data = stability_index
        self.stability_publisher.publish(stability_msg)

        self.get_logger().debug(f'Balance: {balance_state:.3f}, Stability: {stability_index:.3f}')

    def joint_state_callback(self, msg):
        """Process joint state data"""
        self.joint_data = msg

        # Check for joint limits violations
        for i, position in enumerate(msg.position):
            # This is a simplified check - real robots would have specific limits
            if abs(position) > 3.0:  # 3 radians is beyond typical joint limits
                self.get_logger().warn(f'Joint {msg.name[i]} position {position:.3f} exceeds safe limits!')

        # Calculate average joint velocity
        avg_velocity = sum(abs(v) for v in msg.velocity) / len(msg.velocity) if msg.velocity else 0.0
        if avg_velocity > 1.0:  # High velocity warning
            self.get_logger().info(f'High joint velocity detected: {avg_velocity:.3f}')

    def get_sensor_status(self):
        """Return current sensor status"""
        if self.imu_data and self.joint_data:
            return "All sensors operational"
        elif self.imu_data:
            return "IMU operational, joint sensors offline"
        elif self.joint_data:
            return "Joint sensors operational, IMU offline"
        else:
            return "No sensor data received"


def main(args=None):
    rclpy.init(args=args)
    sensor_processor = SensorProcessor()

    try:
        rclpy.spin(sensor_processor)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 3: Create a Launch File

Create a launch file in `sensor_integration_lab/launch/sensor_integration.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # Launch sensor publisher node
        Node(
            package='sensor_integration_lab',
            executable='sensor_publisher',
            name='sensor_publisher',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),

        # Launch sensor processor node
        Node(
            package='sensor_integration_lab',
            executable='sensor_processor',
            name='sensor_processor',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        )
    ])
```

## Step 4: Update Package Configuration

Update the `setup.py` file in your package:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'sensor_integration_lab'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Lab for sensor integration in humanoid robotics',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_publisher = sensor_integration_lab.sensor_publisher:main',
            'sensor_processor = sensor_integration_lab.sensor_processor:main',
        ],
    },
)
```

## Step 5: Build and Run the Package

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select sensor_integration_lab

# Source the workspace
source install/setup.bash

# Run the launch file
ros2 launch sensor_integration_lab sensor_integration.launch.py
```

## Step 6: Testing and Verification

Open a new terminal and run these commands to verify your nodes are working:

```bash
# Check if nodes are running
ros2 node list

# Check topics
ros2 topic list

# Echo sensor data
ros2 topic echo /imu/data

# Echo processed data
ros2 topic echo /balance_state

# Echo joint states
ros2 topic echo /joint_states
```

## Step 7: Using the RAG Chatbot

Now that you have created this lab content, you can test the RAG chatbot integration:

1. Select some text from this lab description
2. Click on the floating AI Assistant button on the textbook page
3. Ask a question related to the selected text, such as:
   - "What does the sensor processor node do?"
   - "How often does the sensor publisher publish data?"
   - "What topics does this lab use for communication?"

The RAG chatbot should provide answers based on the selected text.

## Expected Results

- The sensor publisher node should publish IMU and joint state data at 10 Hz
- The sensor processor node should receive and process this data
- Balance and stability metrics should be published based on the sensor data
- The RAG chatbot should be able to answer questions about this lab when the text is selected

## Troubleshooting

If you encounter issues:

1. **Nodes not found**: Make sure you sourced your workspace (`source install/setup.bash`)
2. **Import errors**: Verify that your Python files have the correct shebang and are executable
3. **Topic issues**: Check that topic names match between publishers and subscribers
4. **Permission errors**: Make sure your Python files are executable (`chmod +x *.py`)

## Extensions

Try these extensions to deepen your understanding:

1. Add a third node that subscribes to the balance_state topic and publishes motor commands
2. Implement a parameter server to configure sensor noise levels
3. Add a service to reset the sensor simulation
4. Create a more complex humanoid model with additional joints

## Summary

In this lab, you've created a complete sensor integration system using ROS 2's publish/subscribe model. You've learned how to:
- Create publisher and subscriber nodes
- Handle different sensor data types
- Process sensor data for robot control
- Use launch files to coordinate multiple nodes
- Test your system with ROS 2 tools

This foundation is essential for building more complex humanoid robotics applications.