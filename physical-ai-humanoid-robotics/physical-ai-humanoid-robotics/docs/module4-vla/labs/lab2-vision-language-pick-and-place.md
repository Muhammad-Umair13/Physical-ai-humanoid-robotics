---
sidebar_label: 'Vision-Language Pick and Place Lab'
sidebar_position: 6
---

# Lab 2: Vision-Language Guided Pick and Place

## Overview

In this lab, you will implement a complete Vision-Language-Action (VLA) system that enables a robotic manipulator to perform pick and place tasks based on natural language commands. This lab integrates computer vision for object detection and localization, natural language processing for command understanding, and action policy generation for robotic control. You'll create a system that can understand commands like "Pick up the red cup from the table" and execute the corresponding pick and place task.

### Learning Objectives

By the end of this lab, you will be able to:
1. Integrate vision and language processing in a unified robotic system
2. Implement spatial reasoning for object localization
3. Create a multimodal policy that combines visual and linguistic inputs
4. Execute complex pick and place tasks using natural language commands
5. Evaluate the performance of VLA systems in real-world scenarios

### Prerequisites

- Completion of the Object Detection lab
- Understanding of ROS 2 action servers and clients
- Basic knowledge of robotic manipulator control
- Familiarity with natural language processing concepts

## Lab Setup

### Environment Preparation

Set up the necessary environment for the vision-language pick and place system:

```bash
# Install additional Python packages needed for this lab
pip install transformers torch torchvision torchaudio openai-clip

# Install ROS 2 manipulation packages
sudo apt install ros-humble-moveit ros-humble-manipulation-msgs
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
```

### Robot Simulation Setup

For this lab, we'll use a simulated robotic manipulator. Set up the simulation environment:

```xml
<!-- launch/vla_pick_place.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directories
    pkg_share = get_package_share_directory('your_robot_package')

    return LaunchDescription([
        # Launch robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': True}
            ]
        ),

        # Launch joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                {'use_sim_time': True}
            ]
        ),

        # Launch vision-language processing node
        Node(
            package='your_robot_package',
            executable='vla_processing_node',
            name='vla_processing',
            parameters=[
                {'use_sim_time': True}
            ]
        ),

        # Launch pick and place action server
        Node(
            package='your_robot_package',
            executable='pick_place_server',
            name='pick_place_server',
            parameters=[
                {'use_sim_time': True}
            ]
        ),

        # Launch command interface
        Node(
            package='your_robot_package',
            executable='command_interface',
            name='command_interface',
            parameters=[
                {'use_sim_time': True}
            ]
        )
    ])
```

## Part 1: Vision-Language Integration

### Multimodal Feature Extraction

First, let's create a node that extracts features from both visual and language inputs:

```python
#!/usr/bin/env python3
"""
Vision-Language Feature Extraction Node
This node extracts and fuses visual and language features
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import clip  # OpenAI CLIP for vision-language features
import numpy as np
from transformers import AutoTokenizer, AutoModel


class VisionLanguageFeatureNode(Node):
    def __init__(self):
        super().__init__('vision_language_feature_node')

        # Initialize CLIP model for vision-language features
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.get_device())

        # Initialize language model for command understanding
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.language_model = AutoModel.from_pretrained("bert-base-uncased")

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Current state
        self.current_image = None
        self.current_command = None

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/robot/command',
            self.command_callback,
            10
        )

        # Create publishers for fused features
        self.feature_pub = self.create_publisher(
            String,  # In practice, you'd use a custom message type
            '/vla/features',
            10
        )

        # Create publisher for object detections with spatial info
        self.spatial_pub = self.create_publisher(
            PointStamped,
            '/object/spatial_location',
            10
        )

        self.get_logger().info('Vision-Language Feature Node initialized')

    def get_device(self):
        """Get appropriate device (CUDA if available)"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def command_callback(self, msg):
        """Process incoming command"""
        self.current_command = msg.data

        # If we have both image and command, process them
        if self.current_image is not None:
            self.process_vision_language_pair()

    def process_vision_language_pair(self):
        """Process the current image and command pair"""
        try:
            # Preprocess image for CLIP
            image_tensor = self.clip_preprocess(self.current_image).unsqueeze(0).to(self.get_device())

            # Tokenize command for language model
            command_tokens = clip.tokenize([self.current_command]).to(self.get_device())

            # Extract visual features using CLIP
            with torch.no_grad():
                visual_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(command_tokens)

            # Normalize features
            visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity between image and text
            similarity = torch.matmul(visual_features, text_features.t())

            # Extract objects from command using simple parsing
            target_objects = self.extract_target_objects(self.current_command)

            # Find relevant objects in image
            object_locations = self.find_objects_in_image(target_objects)

            # Publish fused features
            features_msg = String()
            features_msg.data = str({
                'visual_features': visual_features.cpu().numpy().tolist(),
                'text_features': text_features.cpu().numpy().tolist(),
                'similarity': similarity.cpu().numpy().tolist(),
                'target_objects': target_objects,
                'object_locations': object_locations
            })
            self.feature_pub.publish(features_msg)

            # Publish spatial location of target object
            if object_locations:
                for obj_loc in object_locations:
                    point_msg = PointStamped()
                    point_msg.header.stamp = self.get_clock().now().to_msg()
                    point_msg.header.frame_id = 'camera_rgb_optical_frame'
                    point_msg.point.x = obj_loc['x']
                    point_msg.point.y = obj_loc['y']
                    point_msg.point.z = obj_loc['z']
                    self.spatial_pub.publish(point_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing vision-language pair: {str(e)}')

    def extract_target_objects(self, command):
        """Extract target objects from command using simple parsing"""
        # In practice, use NER or more sophisticated parsing
        objects = ['cup', 'bottle', 'box', 'ball', 'book', 'phone']
        found_objects = []

        command_lower = command.lower()
        for obj in objects:
            if obj in command_lower:
                found_objects.append(obj)

        return found_objects

    def find_objects_in_image(self, target_objects):
        """Find target objects in image using simple color/shape detection"""
        # This is a simplified implementation
        # In practice, use YOLO or other object detection models
        locations = []

        if target_objects:
            # For demonstration, return a fixed location
            # In practice, run object detection on the image
            for obj in target_objects:
                # Simulate finding object at center of image
                h, w, _ = self.current_image.shape
                locations.append({
                    'object': obj,
                    'x': w // 2,
                    'y': h // 2,
                    'confidence': 0.8
                })

        return locations


def main(args=None):
    rclpy.init(args=args)
    node = VisionLanguageFeatureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Advanced Object Detection with Spatial Reasoning

Create a more sophisticated object detection node that includes spatial reasoning:

```python
#!/usr/bin/env python3
"""
Advanced Object Detection with Spatial Reasoning
This node performs object detection and spatial reasoning for pick and place tasks
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped, TransformStamped
from tf2_ros import TransformListener, Buffer
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class SpatialObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('spatial_object_detection_node')

        # Initialize YOLO model
        self.yolo_model = YOLO('yolov8n.pt')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Current state
        self.camera_intrinsics = None

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            Image,  # Use actual CameraInfo message in practice
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Create publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/spatial_detections',
            10
        )

        self.object_pose_pub = self.create_publisher(
            PointStamped,
            '/object/pose',
            10
        )

        self.get_logger().info('Spatial Object Detection Node initialized')

    def camera_info_callback(self, msg):
        """Get camera intrinsics"""
        # In practice, parse CameraInfo message
        # For now, use default values
        self.camera_intrinsics = {
            'fx': 554.256,  # Focal length x
            'fy': 554.256,  # Focal length y
            'cx': 320.5,    # Principal point x
            'cy': 240.5     # Principal point y
        }

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            # Convert ROS depth image to OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.current_depth = depth_image
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {str(e)}')

    def image_callback(self, msg):
        """Process image and perform spatial object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform YOLO detection
            results = self.yolo_model(cv_image, verbose=False)

            # Process detections with spatial information
            spatial_detections = self.process_spatial_detections(
                results, cv_image, self.current_depth if hasattr(self, 'current_depth') else None
            )

            # Publish spatial detections
            detection_array_msg = self.create_spatial_detection_message(spatial_detections, msg.header)
            self.detection_pub.publish(detection_array_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')

    def process_spatial_detections(self, results, image, depth_image):
        """Process detections with spatial information"""
        spatial_detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Get 3D position from depth
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    if depth_image is not None and center_y < depth_image.shape[0] and center_x < depth_image.shape[1]:
                        depth_value = depth_image[center_y, center_x] / 1000.0  # Convert to meters

                        # Convert pixel coordinates to 3D coordinates
                        if depth_value > 0 and self.camera_intrinsics:
                            fx = self.camera_intrinsics['fx']
                            fy = self.camera_intrinsics['fy']
                            cx = self.camera_intrinsics['cx']
                            cy = self.camera_intrinsics['cy']

                            x_3d = (center_x - cx) * depth_value / fx
                            y_3d = (center_y - cy) * depth_value / fy
                            z_3d = depth_value

                            spatial_detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class_id': cls,
                                'class_name': self.yolo_model.names[cls],
                                'position_3d': [x_3d, y_3d, z_3d],
                                'pixel_coords': [center_x, center_y]
                            }

                            spatial_detections.append(spatial_detection)

        return spatial_detections

    def create_spatial_detection_message(self, spatial_detections, header):
        """Create ROS message with spatial detection information"""
        from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose

        detection_array_msg = Detection2DArray()
        detection_array_msg.header = header

        for det in spatial_detections:
            detection_msg = Detection2D()
            detection_msg.header = header

            # Set bounding box
            bbox = BoundingBox2D()
            bbox.size_x = int(det['bbox'][2] - det['bbox'][0])
            bbox.size_y = int(det['bbox'][3] - det['bbox'][1])
            bbox.center.x = int(det['bbox'][0] + bbox.size_x / 2)
            bbox.center.y = int(det['bbox'][1] + bbox.size_y / 2)
            detection_msg.bbox = bbox

            # Set detection result
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det['class_name']
            hypothesis.hypothesis.score = det['confidence']
            detection_msg.results.append(hypothesis)

            # Publish 3D position as separate message
            pose_msg = PointStamped()
            pose_msg.header = header
            pose_msg.point.x = det['position_3d'][0]
            pose_msg.point.y = det['position_3d'][1]
            pose_msg.point.z = det['position_3d'][2]
            self.object_pose_pub.publish(pose_msg)

            detection_array_msg.detections.append(detection_msg)

        return detection_array_msg


def main(args=None):
    rclpy.init(args=args)
    node = SpatialObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 2: Language Understanding for Pick and Place

### Command Parsing and Intent Recognition

Create a node that parses natural language commands and extracts pick and place intents:

```python
#!/usr/bin/env python3
"""
Language Understanding for Pick and Place
This node parses natural language commands for pick and place tasks
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import json
import re


class LanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('language_understanding_node')

        # Create subscriber for commands
        self.command_sub = self.create_subscription(
            String,
            '/robot/command',
            self.command_callback,
            10
        )

        # Create publisher for parsed commands
        self.parsed_command_pub = self.create_publisher(
            String,
            '/parsed/command',
            10
        )

        # Create publisher for target object
        self.target_object_pub = self.create_publisher(
            String,
            '/target/object',
            10
        )

        # Define command patterns
        self.command_patterns = {
            'pick': [
                r'pick up the (.+?) from',
                r'grab the (.+?) from',
                r'take the (.+?) from',
                r'get the (.+?) from'
            ],
            'place': [
                r'to (.+)$',
                r'and place it (?:on|in|at) the (.+)$',
                r'and put it (?:on|in|at) the (.+)$'
            ],
            'location': [
                r'from the (.+?)',
                r'on the (.+?)',
                r'in the (.+?)',
                r'at the (.+?)'
            ]
        }

        self.get_logger().info('Language Understanding Node initialized')

    def command_callback(self, msg):
        """Process incoming natural language command"""
        command = msg.data.lower()

        # Parse the command
        parsed_command = self.parse_command(command)

        # Publish parsed command
        parsed_msg = String()
        parsed_msg.data = json.dumps(parsed_command)
        self.parsed_command_pub.publish(parsed_msg)

        # Publish target object if found
        if 'target_object' in parsed_command:
            target_msg = String()
            target_msg.data = parsed_command['target_object']
            self.target_object_pub.publish(target_msg)

        self.get_logger().info(f'Parsed command: {parsed_command}')

    def parse_command(self, command):
        """Parse natural language command and extract components"""
        result = {
            'action': 'unknown',
            'target_object': None,
            'source_location': None,
            'target_location': None,
            'original_command': command
        }

        # Extract target object and source location
        for pattern in self.command_patterns['pick']:
            match = re.search(pattern, command)
            if match:
                result['target_object'] = match.group(1).strip()
                result['action'] = 'pick_place'
                break

        # Extract source location
        for pattern in self.command_patterns['location']:
            match = re.search(pattern, command)
            if match:
                result['source_location'] = match.group(1).strip()
                break

        # Extract target location (place destination)
        for pattern in self.command_patterns['place']:
            match = re.search(pattern, command)
            if match:
                result['target_location'] = match.group(1).strip()
                break

        # Handle commands without explicit source location
        if result['target_object'] and not result['source_location']:
            # If no source location is specified, assume it's "in front" or "on the table"
            result['source_location'] = 'table'

        # Handle commands without explicit target location
        if result['target_object'] and not result['target_location']:
            # If no target location is specified, assume it's "on the table"
            result['target_location'] = 'table'

        return result


def main(args=None):
    rclpy.init(args=args)
    node = LanguageUnderstandingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 3: Multimodal Action Policy

### Vision-Language-Action Policy Network

Create a neural network that combines visual and language inputs to generate actions:

```python
#!/usr/bin/env python3
"""
Multimodal Action Policy Network
This node implements a neural network that combines vision and language to generate actions
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import numpy as np


class MultimodalPolicyNetwork(nn.Module):
    def __init__(self, vision_feature_dim=512, language_feature_dim=256, action_dim=7):
        super().__init__()

        # Vision feature extractor
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, vision_feature_dim),  # Adjust based on input size
            nn.ReLU()
        )

        # Language feature encoder
        self.language_encoder = nn.Sequential(
            nn.Linear(language_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, vision_feature_dim),
            nn.ReLU()
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vision_feature_dim,
            num_heads=8
        )

        # Action generation
        self.action_head = nn.Sequential(
            nn.Linear(vision_feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, vision_input, language_input):
        # Encode vision
        vision_features = self.vision_encoder(vision_input)

        # Encode language
        language_features = self.language_encoder(language_input)

        # Apply cross-attention
        attended_features, _ = self.cross_attention(
            vision_features.unsqueeze(0),
            language_features.unsqueeze(0),
            language_features.unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)

        # Combine features and generate action
        combined_features = torch.cat([vision_features, attended_features], dim=-1)
        action = self.action_head(combined_features)

        return torch.tanh(action)  # Bound action to [-1, 1]


class VLAActionNode(Node):
    def __init__(self):
        super().__init__('vla_action_node')

        # Initialize policy network
        self.policy_network = MultimodalPolicyNetwork(
            vision_feature_dim=512,
            language_feature_dim=256,
            action_dim=7  # 7-DOF for manipulator
        )

        # Move to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network.to(self.device)
        self.policy_network.eval()

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Current state
        self.current_image = None
        self.current_language_features = None
        self.current_target_object = None

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.feature_sub = self.create_subscription(
            String,
            '/vla/features',
            self.feature_callback,
            10
        )

        self.target_sub = self.create_subscription(
            String,
            '/target/object',
            self.target_callback,
            10
        )

        # Create publisher for actions
        self.action_pub = self.create_publisher(
            Pose,
            '/target/action',
            10
        )

        self.get_logger().info('VLA Action Node initialized')

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            # Convert ROS image to tensor
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(cv_image).float().permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
            image_tensor = image_tensor.to(self.device)

            self.current_image = image_tensor
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def feature_callback(self, msg):
        """Process multimodal features"""
        try:
            features = json.loads(msg.data)
            # Extract language features (simplified)
            if 'text_features' in features:
                lang_features = np.array(features['text_features'])
                self.current_language_features = torch.from_numpy(lang_features).float().to(self.device)
        except Exception as e:
            self.get_logger().error(f'Error processing features: {str(e)}')

    def target_callback(self, msg):
        """Process target object"""
        self.current_target_object = msg.data

    def generate_action(self):
        """Generate action using VLA policy"""
        if (self.current_image is not None and
            self.current_language_features is not None):

            try:
                with torch.no_grad():
                    action = self.policy_network(
                        self.current_image,
                        self.current_language_features.unsqueeze(0)
                    )

                # Convert action to Pose message (simplified)
                action_msg = Pose()
                action_msg.position.x = float(action[0, 0])
                action_msg.position.y = float(action[0, 1])
                action_msg.position.z = float(action[0, 2])
                action_msg.orientation.x = float(action[0, 3])
                action_msg.orientation.y = float(action[0, 4])
                action_msg.orientation.z = float(action[0, 5])
                action_msg.orientation.w = float(action[0, 6])

                self.action_pub.publish(action_msg)

                self.get_logger().info(f'Generated action for target: {self.current_target_object}')

            except Exception as e:
                self.get_logger().error(f'Error generating action: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = VLAActionNode()

    # Create timer to periodically generate actions
    node.create_timer(0.1, node.generate_action)  # 10Hz

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 4: Pick and Place Action Server

### Complete Pick and Place Implementation

Create an action server that executes the complete pick and place task:

```python
#!/usr/bin/env python3
"""
Pick and Place Action Server
This server executes complete pick and place tasks based on VLA inputs
"""

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import String
from moveit_msgs.action import MoveGroup
from control_msgs.action import FollowJointTrajectory
import threading
import time
import json


class PickPlaceActionServer(Node):
    def __init__(self):
        super().__init__('pick_place_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            MoveGroup,  # Using MoveGroup action for pick and place
            'pick_place_task',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Publishers for monitoring
        self.status_pub = self.create_publisher(String, '/pick_place/status', 10)
        self.command_pub = self.create_publisher(String, '/robot/command', 10)

        # Current state
        self.current_target = None
        self.current_location = None

        self.get_logger().info('Pick and Place Action Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject goal request"""
        self.get_logger().info('Received pick and place goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the pick and place task"""
        self.get_logger().info('Executing pick and place task...')

        feedback_msg = MoveGroup.Feedback()
        result = MoveGroup.Result()

        # Parse the goal command
        command = goal_handle.request
        parsed_command = self.parse_command(command)

        try:
            # Step 1: Navigate to object location
            self.get_logger().info('Step 1: Navigating to object location')
            feedback_msg.feedback = "Navigating to object location"
            goal_handle.publish_feedback(feedback_msg)

            if not self.navigate_to_object(parsed_command['target_object']):
                goal_handle.abort()
                result.error_code = -1
                return result

            # Step 2: Detect and locate object
            self.get_logger().info('Step 2: Detecting object')
            feedback_msg.feedback = "Detecting object"
            goal_handle.publish_feedback(feedback_msg)

            object_pose = self.detect_object(parsed_command['target_object'])
            if not object_pose:
                goal_handle.abort()
                result.error_code = -2
                return result

            # Step 3: Approach object
            self.get_logger().info('Step 3: Approaching object')
            feedback_msg.feedback = "Approaching object"
            goal_handle.publish_feedback(feedback_msg)

            if not self.approach_object(object_pose):
                goal_handle.abort()
                result.error_code = -3
                return result

            # Step 4: Grasp object
            self.get_logger().info('Step 4: Grasping object')
            feedback_msg.feedback = "Grasping object"
            goal_handle.publish_feedback(feedback_msg)

            if not self.grasp_object():
                goal_handle.abort()
                result.error_code = -4
                return result

            # Step 5: Lift object
            self.get_logger().info('Step 5: Lifting object')
            feedback_msg.feedback = "Lifting object"
            goal_handle.publish_feedback(feedback_msg)

            if not self.lift_object():
                goal_handle.abort()
                result.error_code = -5
                return result

            # Step 6: Navigate to destination
            self.get_logger().info('Step 6: Navigating to destination')
            feedback_msg.feedback = "Navigating to destination"
            goal_handle.publish_feedback(feedback_msg)

            if not self.navigate_to_destination(parsed_command['target_location']):
                goal_handle.abort()
                result.error_code = -6
                return result

            # Step 7: Place object
            self.get_logger().info('Step 7: Placing object')
            feedback_msg.feedback = "Placing object"
            goal_handle.publish_feedback(feedback_msg)

            if not self.place_object():
                goal_handle.abort()
                result.error_code = -7
                return result

            # Step 8: Retract gripper
            self.get_logger().info('Step 8: Retracting gripper')
            feedback_msg.feedback = "Retracting gripper"
            goal_handle.publish_feedback(feedback_msg)

            if not self.retract_gripper():
                goal_handle.abort()
                result.error_code = -8
                return result

            # Task completed successfully
            goal_handle.succeed()
            result.error_code = 1  # SUCCESS
            self.get_logger().info('Pick and place task completed successfully!')

        except Exception as e:
            self.get_logger().error(f'Error during pick and place execution: {str(e)}')
            goal_handle.abort()
            result.error_code = -9  # GENERAL ERROR

        return result

    def parse_command(self, command):
        """Parse command to extract target object and destination"""
        # In a real implementation, this would be more sophisticated
        # For this example, we'll use simple parsing
        return {
            'target_object': 'cup',  # This would come from VLA system
            'target_location': 'table'
        }

    def navigate_to_object(self, target_object):
        """Navigate the robot base to the object location"""
        # Simulate navigation
        self.get_logger().info(f'Navigating to {target_object} location')
        time.sleep(2)  # Simulate navigation time
        return True

    def detect_object(self, target_object):
        """Detect the target object using vision system"""
        # In a real implementation, this would interface with the vision system
        # For simulation, return a dummy pose
        self.get_logger().info(f'Detecting {target_object}')
        time.sleep(1)  # Simulate detection time

        # Return dummy pose (in practice, this comes from vision system)
        return Pose(
            position=Point(x=0.5, y=0.0, z=0.2),
            orientation=Point(x=0.0, y=0.0, z=0.0, w=1.0)
        )

    def approach_object(self, object_pose):
        """Approach the detected object"""
        self.get_logger().info('Approaching object')
        time.sleep(2)  # Simulate approach time
        return True

    def grasp_object(self):
        """Grasp the object using the robot gripper"""
        self.get_logger().info('Grasping object')
        time.sleep(1)  # Simulate grasp time
        return True

    def lift_object(self):
        """Lift the object after grasping"""
        self.get_logger().info('Lifting object')
        time.sleep(1)  # Simulate lift time
        return True

    def navigate_to_destination(self, destination):
        """Navigate to the destination location"""
        self.get_logger().info(f'Navigating to {destination}')
        time.sleep(2)  # Simulate navigation time
        return True

    def place_object(self):
        """Place the object at the destination"""
        self.get_logger().info('Placing object')
        time.sleep(1)  # Simulate place time
        return True

    def retract_gripper(self):
        """Retract the gripper after placing"""
        self.get_logger().info('Retracting gripper')
        time.sleep(1)  # Simulate retraction time
        return True


def main(args=None):
    rclpy.init(args=args)

    # Create the action server node
    pick_place_server = PickPlaceActionServer()

    # Use multi-threaded executor to handle callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(pick_place_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pick_place_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 5: Command Interface

### Natural Language Command Interface

Create a simple command interface that allows users to input natural language commands:

```python
#!/usr/bin/env python3
"""
Vision-Language Command Interface
This node provides a command interface for the VLA pick and place system
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
import time


class CommandInterfaceNode(Node):
    def __init__(self):
        super().__init__('command_interface_node')

        # Create publisher for commands
        self.command_pub = self.create_publisher(String, '/robot/command', 10)

        # Create action client for pick and place
        self.pick_place_client = ActionClient(self, MoveGroup, 'pick_place_task')

        # Wait for action server
        self.get_logger().info('Waiting for pick and place action server...')
        self.pick_place_client.wait_for_server()

        # Start command input loop in a separate thread
        self.command_thread = threading.Thread(target=self.command_input_loop)
        self.command_thread.daemon = True
        self.command_thread.start()

        self.get_logger().info('Command Interface Node initialized')

    def command_input_loop(self):
        """Loop to accept user commands"""
        while rclpy.ok():
            try:
                # Get command from user
                command = input("\nEnter command (or 'quit' to exit): ")

                if command.lower() == 'quit':
                    break

                # Publish command
                command_msg = String()
                command_msg.data = command
                self.command_pub.publish(command_msg)

                self.get_logger().info(f'Sent command: {command}')

                # Wait a bit for processing
                time.sleep(0.1)

            except EOFError:
                # Handle Ctrl+D
                break
            except Exception as e:
                self.get_logger().error(f'Error in command input: {str(e)}')

    def send_pick_place_goal(self, command):
        """Send goal to pick and place action server"""
        goal_msg = MoveGroup.Goal()
        # In a real implementation, this would include the parsed command details
        goal_msg.request.group_name = 'manipulator'  # Robot group name

        self.get_logger().info('Sending pick and place goal...')

        # Send goal and wait for result
        future = self.pick_place_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        return future

    def feedback_callback(self, feedback_msg):
        """Handle feedback from action server"""
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')


def main(args=None):
    rclpy.init(args=args)
    node = CommandInterfaceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 6: Complete System Integration

### Main Integration Node

Create a main node that orchestrates all components:

```python
#!/usr/bin/env python3
"""
Vision-Language-Action Integration Node
Main node that orchestrates the complete VLA pick and place system
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import json
import time


class VLAPickPlaceNode(Node):
    def __init__(self):
        super().__init__('vla_pick_place_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # System state
        self.current_command = None
        self.current_image = None
        self.current_detections = []
        self.current_target_object = None
        self.current_object_pose = None

        # Create subscribers
        self.command_sub = self.create_subscription(
            String,
            '/robot/command',
            self.command_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            String,
            '/parsed/command',
            self.parsed_command_callback,
            10
        )

        self.spatial_sub = self.create_subscription(
            PointStamped,
            '/object/spatial_location',
            self.spatial_callback,
            10
        )

        # Create publishers
        self.status_pub = self.create_publisher(String, '/vla/status', 10)
        self.action_pub = self.create_publisher(String, '/vla/action', 10)

        self.get_logger().info('VLA Pick Place Node initialized')

    def command_callback(self, msg):
        """Handle incoming command"""
        self.current_command = msg.data
        self.get_logger().info(f'Received command: {self.current_command}')
        self.process_command()

    def image_callback(self, msg):
        """Handle incoming image"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def parsed_command_callback(self, msg):
        """Handle parsed command"""
        try:
            parsed = json.loads(msg.data)
            self.current_target_object = parsed.get('target_object')
            self.get_logger().info(f'Parsed target object: {self.current_target_object}')
        except Exception as e:
            self.get_logger().error(f'Error parsing command: {str(e)}')

    def spatial_callback(self, msg):
        """Handle spatial information"""
        self.current_object_pose = msg.point
        self.get_logger().info(f'Object pose: ({msg.point.x}, {msg.point.y}, {msg.point.z})')

    def process_command(self):
        """Process the current command and coordinate system components"""
        if not self.current_command:
            return

        # Publish status
        status_msg = String()
        status_msg.data = f'Processing command: {self.current_command}'
        self.status_pub.publish(status_msg)

        # Coordinate the vision-language-action pipeline
        self.get_logger().info('Starting VLA pipeline...')

        # The actual pipeline would be coordinated here
        # In a real system, this would trigger the appropriate sequence of events
        # 1. Vision system detects objects
        # 2. Language system parses command
        # 3. Action system generates and executes plan

        # For this example, we'll just log the process
        self.get_logger().info('VLA pipeline completed')

    def execute_pick_place(self):
        """Execute the complete pick and place task"""
        # This would coordinate with the action server
        # In practice, this would send a goal to the pick_place_action_server
        pass


def main(args=None):
    rclpy.init(args=args)
    node = VLAPickPlaceNode()

    # Run the node
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Lab Assessment

### Assessment Tasks

Complete the following tasks to demonstrate your understanding:

1. **System Integration**:
   - Successfully integrate all components (vision, language, action)
   - Verify that the system can process a natural language command
   - Confirm that object detection works with spatial reasoning

2. **Command Processing**:
   - Test the system with various natural language commands
   - Verify that commands like "Pick up the red cup from the table" are correctly parsed
   - Check that the system identifies the correct target object

3. **Action Execution**:
   - If using simulation, verify that the robot can execute pick and place actions
   - If using real hardware, ensure safety protocols are in place
   - Evaluate the success rate of pick and place operations

4. **Performance Evaluation**:
   - Measure the system's response time from command input to action execution
   - Evaluate the accuracy of object detection and localization
   - Document any limitations or failure cases

### Questions for Reflection

1. How does the integration of vision and language improve the robot's ability to perform pick and place tasks?
2. What are the main challenges in combining visual and linguistic information for robotic control?
3. How would you improve the system's ability to handle ambiguous commands?
4. What safety considerations are important when implementing VLA systems for physical robots?
5. How could you extend this system to handle more complex manipulation tasks?

## Summary

In this lab, you've implemented a complete Vision-Language-Action system for robotic pick and place tasks. You learned how to:
- Integrate vision and language processing in a unified robotic system
- Implement spatial reasoning for object localization
- Create a multimodal policy that combines visual and linguistic inputs
- Execute complex pick and place tasks using natural language commands
- Coordinate multiple ROS 2 nodes to achieve a complex robotic task

The VLA system you've built demonstrates the power of combining multiple AI modalities for robotic applications. By integrating visual perception, natural language understanding, and action generation, robots can perform complex tasks that require understanding both the environment and human intentions expressed in natural language.

This system forms the foundation for more advanced robotic applications where robots can understand and execute complex, natural language commands in dynamic environments, making them more accessible and useful for everyday tasks.