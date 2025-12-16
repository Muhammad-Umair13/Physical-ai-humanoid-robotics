---
sidebar_label: 'Lab 2: Unity-ROS 2 Bridge for Humanoid Robot Control'
sidebar_position: 2
---

# Lab 2: Unity-ROS 2 Bridge for Humanoid Robot Control

## Objective

In this lab, you will create a Unity scene with a humanoid robot model and establish communication with ROS 2 using the ROS TCP Connector. You will learn to publish sensor data from Unity to ROS 2 and subscribe to control commands from ROS 2 to control the Unity robot.

## Prerequisites

- Unity Hub and Unity 2021.3 LTS or later installed
- ROS 2 Humble Hawksbill installed
- Basic knowledge of C# programming
- Understanding of ROS 2 concepts (topics, messages, publishers, subscribers)

## Learning Outcomes

By the end of this lab, you will be able to:
1. Set up the ROS TCP Connector in Unity
2. Create a humanoid robot model in Unity
3. Publish sensor data from Unity to ROS 2
4. Subscribe to ROS 2 topics to control the Unity robot
5. Implement bidirectional communication between Unity and ROS 2

## Step 1: Setting Up Unity Project

Create a new Unity 3D project named "UnityROS2Humanoid" and install the required packages:

1. Open Unity Hub and create a new 3D project
2. Go to Window → Package Manager
3. Install the following packages:
   - ROS TCP Connector
   - Unity Robotics Tools (optional but helpful)

## Step 2: Creating the Humanoid Robot Model

Create a simple humanoid robot using Unity's GameObject hierarchy. Create a C# script called `HumanoidRobot.cs`:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;
using System.Collections.Generic;

public class HumanoidRobot : MonoBehaviour
{
    [Header("Robot Configuration")]
    public List<Transform> jointTransforms = new List<Transform>();
    public List<string> jointNames = new List<string>();

    [Header("Sensors")]
    public Camera cameraSensor;
    public Transform imuTransform;

    [Header("ROS Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    private ROSConnection ros;
    private int jointCount;

    // Publishers
    private string jointStatesTopic = "/joint_states";
    private string cameraTopic = "/camera/image_raw";
    private string imuTopic = "/imu/data";

    // Subscribers
    private string jointCommandsTopic = "/joint_group_position_controller/command";

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIPAddress, rosPort);

        // Register publishers
        ros.RegisterPublisher<JointStateMsg>(jointStatesTopic);
        ros.RegisterPublisher<ImageMsg>(cameraTopic);
        ros.RegisterPublisher<ImuMsg>(imuTopic);

        // Register subscribers
        ros.RegisterSubscriber<std_msgs.Float64MultiArray>(jointCommandsTopic, OnJointCommandsReceived);

        jointCount = jointTransforms.Count;

        // Set up camera sensor
        if (cameraSensor != null)
        {
            cameraSensor.enabled = true;
            cameraSensor.targetTexture = new RenderTexture(640, 480, 24);
        }
    }

    void Update()
    {
        // Publish sensor data periodically
        if (Time.time % 0.1f < Time.deltaTime) // Every 0.1 seconds
        {
            PublishJointStates();
            PublishCameraData();
            PublishIMUData();
        }
    }

    void PublishJointStates()
    {
        JointStateMsg jointState = new JointStateMsg();
        jointState.header = new std_msgs.Header();
        jointState.header.stamp = new builtin_interfaces.Time();
        jointState.header.frame_id = "base_link";

        jointState.name = new List<string>(jointNames.ToArray());
        jointState.position = new List<double>();
        jointState.velocity = new List<double>();
        jointState.effort = new List<double>();

        for (int i = 0; i < jointCount; i++)
        {
            // Convert Unity rotation (degrees) to radians for ROS
            float angleInRadians = jointTransforms[i].localEulerAngles.y * Mathf.Deg2Rad;
            jointState.position.Add(angleInRadians);
            jointState.velocity.Add(0.0); // Placeholder
            jointState.effort.Add(0.0);   // Placeholder
        }

        ros.Publish(jointStatesTopic, jointState);
    }

    void PublishCameraData()
    {
        if (cameraSensor == null) return;

        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = cameraSensor.targetTexture;

        Texture2D image = new Texture2D(cameraSensor.targetTexture.width,
                                       cameraSensor.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, cameraSensor.targetTexture.width,
                                 cameraSensor.targetTexture.height), 0, 0);
        image.Apply();

        RenderTexture.active = currentRT;

        ImageMsg rosImage = new ImageMsg();
        rosImage.header = new std_msgs.Header();
        rosImage.header.stamp = new builtin_interfaces.Time();
        rosImage.header.frame_id = "camera_link";
        rosImage.height = (uint)cameraSensor.targetTexture.height;
        rosImage.width = (uint)cameraSensor.targetTexture.width;
        rosImage.encoding = "rgb8";
        rosImage.is_bigendian = 0;
        rosImage.step = (uint)(3 * cameraSensor.targetTexture.width);

        // Convert texture to bytes (simplified - in practice you'd do proper conversion)
        rosImage.data = System.Text.Encoding.ASCII.GetBytes("dummy_data");

        ros.Publish(cameraTopic, rosImage);
    }

    void PublishIMUData()
    {
        ImuMsg imuMsg = new ImuMsg();
        imuMsg.header = new std_msgs.Header();
        imuMsg.header.stamp = new builtin_interfaces.Time();
        imuMsg.header.frame_id = "imu_link";

        // Set orientation (simplified)
        imuMsg.orientation = new geometry_msgs.Quaternion(
            imuTransform.rotation.x,
            imuTransform.rotation.y,
            imuTransform.rotation.z,
            imuTransform.rotation.w
        );

        // Set angular velocity (simplified)
        imuMsg.angular_velocity = new geometry_msgs.Vector3(0, 0, 0);

        // Set linear acceleration (simplified)
        imuMsg.linear_acceleration = new geometry_msgs.Vector3(0, 0, 9.81f);

        ros.Publish(imuTopic, imuMsg);
    }

    void OnJointCommandsReceived(std_msgs.Float64MultiArray msg)
    {
        // Process joint commands from ROS 2
        if (msg.data.Count == jointCount)
        {
            for (int i = 0; i < jointCount; i++)
            {
                // Convert radians to degrees for Unity
                float targetAngle = (float)msg.data[i] * Mathf.Rad2Deg;

                // Apply the joint rotation
                Vector3 currentRotation = jointTransforms[i].localEulerAngles;
                jointTransforms[i].localRotation = Quaternion.Euler(currentRotation.x, targetAngle, currentRotation.z);
            }
        }
    }
}
```

## Step 3: Setting Up the Unity Scene

1. Create an empty GameObject named "RobotBase" and attach the `HumanoidRobot` script to it
2. Create the robot hierarchy with the following structure:

```
RobotBase (with HumanoidRobot script)
├── Torso
│   ├── Head
│   ├── LeftUpperLeg
│   ├── RightUpperLeg
│   ├── LeftUpperArm
│   └── RightUpperArm
├── LeftLowerLeg
├── RightLowerLeg
├── LeftLowerArm
├── RightLowerArm
├── LeftFoot
├── RightFoot
├── LeftHand
└── RightHand
```

3. Assign the joint transforms to the `jointTransforms` list in the HumanoidRobot script
4. Assign corresponding joint names to the `jointNames` list:
   - "left_hip_joint", "right_hip_joint", "left_knee_joint", "right_knee_joint"
   - "left_ankle_joint", "right_ankle_joint", "left_shoulder_joint", "right_shoulder_joint"
   - "left_elbow_joint", "right_elbow_joint"

## Step 4: Creating a Camera Sensor

1. Create a Camera as a child of the robot head
2. Tag it as the camera sensor in the HumanoidRobot script
3. Make sure the camera has a target texture assigned

## Step 5: Creating ROS 2 Control Node

Create a ROS 2 package for controlling the Unity robot. Create the file `unity_robot_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image, Imu
from geometry_msgs.msg import Twist
import time
import math

class UnityRobotController(Node):
    def __init__(self):
        super().__init__('unity_robot_controller')

        # Publisher for joint commands
        self.joint_cmd_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/command',
            10
        )

        # Subscribers for sensor data
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.camera_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Timer for sending commands
        self.timer = self.create_timer(0.1, self.send_commands)

        self.joint_positions = [0.0] * 10  # Assuming 10 joints
        self.command_counter = 0

        self.get_logger().info('Unity Robot Controller initialized')

    def joint_state_callback(self, msg):
        """Callback for receiving joint states from Unity"""
        self.joint_positions = list(msg.position)
        self.get_logger().debug(f'Joint positions: {self.joint_positions}')

    def camera_callback(self, msg):
        """Callback for receiving camera images from Unity"""
        # Process camera data (in a real application, you'd process the image)
        self.get_logger().debug(f'Camera image received: {msg.width}x{msg.height}')

    def imu_callback(self, msg):
        """Callback for receiving IMU data from Unity"""
        orientation = msg.orientation
        self.get_logger().debug(f'IMU orientation: ({orientation.x}, {orientation.y}, {orientation.z}, {orientation.w})')

    def send_commands(self):
        """Send joint position commands to Unity"""
        cmd_msg = Float64MultiArray()

        # Create a walking gait pattern
        t = self.command_counter * 0.1  # Time in seconds

        # Simple walking pattern for humanoid
        positions = []

        # Hip joints - opposite movement for walking
        positions.append(math.sin(t) * 0.2)    # left_hip
        positions.append(math.sin(t + math.pi) * 0.2)  # right_hip

        # Knee joints - follow hip with phase offset
        positions.append(math.sin(t + math.pi/2) * 0.3)  # left_knee
        positions.append(math.sin(t + math.pi + math.pi/2) * 0.3)  # right_knee

        # Ankle joints - smaller movement
        positions.append(math.sin(t + math.pi/4) * 0.1)  # left_ankle
        positions.append(math.sin(t + math.pi + math.pi/4) * 0.1)  # right_ankle

        # Arm joints - counterbalance
        positions.append(math.sin(t + math.pi) * 0.3)    # left_shoulder
        positions.append(math.sin(t) * 0.3)              # right_shoulder

        positions.append(math.sin(t + math.pi/3) * 0.2)  # left_elbow
        positions.append(math.sin(t + math.pi + math.pi/3) * 0.2)  # right_elbow

        cmd_msg.data = positions

        self.joint_cmd_publisher.publish(cmd_msg)
        self.command_counter += 1

def main(args=None):
    rclpy.init(args=args)

    controller = UnityRobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 6: Creating a More Advanced Robot Controller

Create a more sophisticated controller in ROS 2 called `advanced_humanoid_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np
import math

class AdvancedHumanoidController(Node):
    def __init__(self):
        super().__init__('advanced_humanoid_controller')

        # Publishers
        self.joint_cmd_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/command',
            10
        )

        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Subscribers
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Command subscribers
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel_input',
            self.cmd_vel_input_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # Robot state
        self.current_joint_positions = np.zeros(10)
        self.desired_joint_positions = np.zeros(10)
        self.imu_data = None
        self.cmd_vel_input = Twist()

        # Walking gait parameters
        self.gait_phase = 0.0
        self.step_frequency = 1.0  # Hz
        self.step_amplitude = 0.3  # radians

        self.get_logger().info('Advanced Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        if len(msg.position) >= 10:
            self.current_joint_positions = np.array(msg.position[:10])

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        self.imu_data = msg

    def cmd_vel_input_callback(self, msg):
        """Receive velocity commands"""
        self.cmd_vel_input = msg

    def control_loop(self):
        """Main control loop"""
        # Update gait phase based on time
        self.gait_phase += 2 * math.pi * self.step_frequency * 0.05

        # Generate walking pattern based on velocity input
        desired_positions = self.generate_walking_pattern()

        # Publish joint commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = desired_positions.tolist()
        self.joint_cmd_publisher.publish(cmd_msg)

        # Optionally publish as trajectory
        self.publish_trajectory(desired_positions)

    def generate_walking_pattern(self):
        """Generate joint positions for walking"""
        positions = np.zeros(10)

        # Calculate walking parameters based on input velocity
        forward_speed = self.cmd_vel_input.linear.x
        turn_speed = self.cmd_vel_input.angular.z

        # Basic walking gait
        left_leg_phase = self.gait_phase
        right_leg_phase = self.gait_phase + math.pi  # Opposite phase

        # Hip joints
        positions[0] = math.sin(left_leg_phase) * self.step_amplitude * forward_speed  # left_hip
        positions[1] = math.sin(right_leg_phase) * self.step_amplitude * forward_speed  # right_hip

        # Knee joints (with phase offset for natural walking)
        positions[2] = math.sin(left_leg_phase + math.pi/2) * self.step_amplitude * forward_speed  # left_knee
        positions[3] = math.sin(right_leg_phase + math.pi/2) * self.step_amplitude * forward_speed  # right_knee

        # Ankle joints (smaller movement for balance)
        positions[4] = math.sin(left_leg_phase - math.pi/4) * 0.1 * forward_speed  # left_ankle
        positions[5] = math.sin(right_leg_phase - math.pi/4) * 0.1 * forward_speed  # right_ankle

        # Arm joints (counterbalance to leg movement)
        positions[6] = -math.sin(left_leg_phase + math.pi) * 0.2 * forward_speed  # left_shoulder
        positions[7] = -math.sin(right_leg_phase + math.pi) * 0.2 * forward_speed  # right_shoulder

        positions[8] = math.sin(left_leg_phase + math.pi/3) * 0.15 * forward_speed  # left_elbow
        positions[9] = math.sin(right_leg_phase + math.pi/3) * 0.15 * forward_speed  # right_elbow

        # Add turning component
        positions[6] += turn_speed * 0.3  # Left arm counter-turn
        positions[7] -= turn_speed * 0.3  # Right arm counter-turn

        return positions

    def publish_trajectory(self, positions):
        """Publish joint trajectory message"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [
            'left_hip_joint', 'right_hip_joint',
            'left_knee_joint', 'right_knee_joint',
            'left_ankle_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'right_shoulder_joint',
            'left_elbow_joint', 'right_elbow_joint'
        ]

        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.velocities = [0.0] * len(positions)  # Zero velocities for simplicity
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50000000  # 50ms

        traj_msg.points = [point]
        self.trajectory_publisher.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)

    controller = AdvancedHumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 7: Creating a Launch File

Create a ROS 2 launch file `unity_humanoid_bringup.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    unity_ip = DeclareLaunchArgument(
        'unity_ip',
        default_value='127.0.0.1',
        description='IP address of the Unity application'
    )

    unity_port = DeclareLaunchArgument(
        'unity_port',
        default_value='10000',
        description='Port for Unity-ROS communication'
    )

    # Unity controller node
    unity_controller = Node(
        package='your_package_name',  # Replace with your actual package name
        executable='unity_robot_controller',
        name='unity_robot_controller',
        parameters=[
            {'unity_ip': LaunchConfiguration('unity_ip')},
            {'unity_port': LaunchConfiguration('unity_port')}
        ],
        output='screen'
    )

    # Advanced humanoid controller
    humanoid_controller = Node(
        package='your_package_name',  # Replace with your actual package name
        executable='advanced_humanoid_controller',
        name='advanced_humanoid_controller',
        output='screen'
    )

    # Joint state publisher (optional)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'rate': 50}]
    )

    # Robot state publisher (optional)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'rate': 50}]
    )

    return LaunchDescription([
        unity_ip,
        unity_port,
        unity_controller,
        humanoid_controller,
        joint_state_publisher,
        robot_state_publisher
    ])
```

## Step 8: Running the Simulation

1. First, make sure ROS 2 is sourced:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash  # Your ROS workspace
```

2. Start the ROS 2 nodes:
```bash
# Terminal 1: Start the controllers
ros2 run your_package_name unity_robot_controller

# Terminal 2: Or use the launch file
ros2 launch your_package_name unity_humanoid_bringup.launch.py
```

3. In Unity, make sure the ROS IP address is set to your machine's IP (or 127.0.0.1 if ROS is running locally)

4. Press Play in Unity to start the simulation

## Step 9: Testing the Integration

### Basic Communication Test
1. Verify that joint states are being published from Unity to ROS 2
2. Check that camera and IMU data are being transmitted
3. Confirm that joint commands from ROS 2 are affecting the Unity robot

### Walking Pattern Test
1. Use the advanced controller to send walking commands
2. Observe the robot's gait in Unity
3. Adjust parameters to improve the walking pattern

### Sensor Data Validation
1. Monitor the sensor topics in ROS 2
2. Verify that data is being published at the expected rate
3. Check that sensor values are reasonable

## Step 10: Troubleshooting Common Issues

### Connection Issues
```bash
# Check if the port is accessible
telnet 127.0.0.1 10000

# Check ROS 2 nodes
ros2 node list
ros2 topic list

# Check topic echo
ros2 topic echo /joint_states
```

### Performance Issues
- Reduce the publishing rate in Unity
- Simplify the robot model if needed
- Check Unity's frame rate

### Synchronization Issues
- Ensure time synchronization between Unity and ROS 2
- Check that message timestamps are properly set
- Verify that the control loop rates are appropriate

## Expected Results

After completing this lab, you should have:
- A Unity scene with a humanoid robot model
- Working ROS TCP connection between Unity and ROS 2
- Bidirectional communication for sensor data and control commands
- A ROS 2 node that can control the Unity robot
- Understanding of how to integrate Unity simulations with ROS 2

## Extensions

Try these extensions to enhance your understanding:

1. Add more sophisticated control algorithms (PID controllers)
2. Implement balance control using IMU feedback
3. Add computer vision processing to the camera feed
4. Create more complex humanoid behaviors
5. Implement state machines for different robot behaviors
6. Add physics-based simulation in Unity with realistic constraints

## Summary

In this lab, you've successfully created a Unity-ROS 2 bridge for humanoid robot control. You've learned how to set up communication channels, publish sensor data from Unity, subscribe to control commands from ROS 2, and implement basic humanoid locomotion patterns. This integration forms the foundation for more advanced robotics simulation and control applications.