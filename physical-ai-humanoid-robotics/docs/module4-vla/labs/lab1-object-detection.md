---
sidebar_label: 'Object Detection Lab'
sidebar_position: 5
---

# Lab 1: Object Detection with YOLO for Robotics

## Overview

In this lab, you will implement and deploy a YOLO-based object detection system for robotic applications. You'll learn how to integrate YOLO models with ROS 2, optimize them for real-time performance, and use the detection results to guide robotic actions. This lab will provide hands-on experience with the vision component of Vision-Language-Action (VLA) systems.

### Learning Objectives

By the end of this lab, you will be able to:
1. Set up and configure YOLO models for robotic applications
2. Integrate YOLO with ROS 2 for real-time object detection
3. Optimize YOLO performance using NVIDIA TensorRT
4. Use detection results to trigger robotic actions
5. Evaluate detection accuracy and performance metrics

### Prerequisites

- Basic understanding of ROS 2 concepts
- Python programming experience
- Familiarity with computer vision concepts
- NVIDIA GPU with CUDA support (for TensorRT optimization)

## Lab Setup

### Environment Preparation

First, let's set up the necessary environment for this lab:

```bash
# Create a new ROS 2 workspace for the lab
mkdir -p ~/vla_ws/src
cd ~/vla_ws

# Install required Python packages
pip install ultralytics opencv-python torch torchvision tensorrt pycuda

# Install ROS 2 dependencies
sudo apt update
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-image-transport
```

### YOLO Model Setup

For this lab, we'll use YOLOv8, which provides excellent performance for robotic applications:

```python
# Install YOLOv8
pip install ultralytics

# Download a pre-trained YOLOv8 model
from ultralytics import YOLO

# Download and save the model
model = YOLO('yolov8n.pt')  # nano version for faster inference
model.save('yolov8n_robotics.pt')
```

## Part 1: Basic YOLO Implementation

### Simple Object Detection Node

Let's start by creating a basic YOLO object detection node:

```python
#!/usr/bin/env python3
"""
Basic YOLO Object Detection Node
This node subscribes to camera images and publishes detection results
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch


class YOLODetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')

        # Initialize YOLO model
        self.yolo_model = YOLO('yolov8n.pt')

        # Set device based on availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model.to(self.device)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Create publisher for detection results
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )

        self.get_logger().info('YOLO Detection Node initialized')

    def image_callback(self, msg):
        """Process incoming image and perform object detection"""
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform YOLO inference
            results = self.yolo_model(cv_image, verbose=False)

            # Convert results to ROS message
            detection_array_msg = self.process_detections(results, msg.header)

            # Publish detections
            self.detection_pub.publish(detection_array_msg)

            self.get_logger().info(f'Published {len(detection_array_msg.detections)} detections')

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')

    def process_detections(self, results, header):
        """Process YOLO results and convert to ROS message"""
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = header

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Create detection message
                    detection_msg = Detection2D()
                    detection_msg.header = header

                    # Set bounding box
                    detection_msg.bbox.size_x = int(x2 - x1)
                    detection_msg.bbox.size_y = int(y2 - y1)
                    detection_msg.bbox.center.x = int(x1 + (x2 - x1) / 2)
                    detection_msg.bbox.center.y = int(y1 + (y2 - y1) / 2)

                    # Set detection result
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = self.yolo_model.names[cls]
                    hypothesis.hypothesis.score = float(conf)
                    detection_msg.results.append(hypothesis)

                    detection_array_msg.detections.append(detection_msg)

        return detection_array_msg


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Launch File

Create a launch file to run the detection node:

```xml
<!-- yolov8_detection.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='your_robot_package',
            executable='yolo_detection_node',
            name='yolo_detection',
            parameters=[
                {'use_sim_time': False}
            ],
            output='screen'
        )
    ])
```

## Part 2: Performance Optimization with TensorRT

### TensorRT Optimization

For real-time performance on NVIDIA hardware, we'll optimize the YOLO model using TensorRT:

```python
#!/usr/bin/env python3
"""
TensorRT Optimized YOLO Node
This node uses TensorRT optimized YOLO for improved performance
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTYOLONode(Node):
    def __init__(self):
        super().__init__('tensorrt_yolo_node')

        # Initialize TensorRT engine
        self.engine = self.load_engine('/path/to/yolov8n.engine')
        self.context = self.engine.create_execution_context()

        # Initialize buffers
        self.initialize_buffers()

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create subscriber and publisher
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )

        self.get_logger().info('TensorRT YOLO Node initialized')

    def load_engine(self, engine_path):
        """Load TensorRT engine"""
        with open(engine_path, 'rb') as f:
            return trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

    def initialize_buffers(self):
        """Initialize input and output buffers for TensorRT"""
        # Get input and output binding info
        self.input_binding = self.engine.get_binding_index('images')
        self.output_binding = self.engine.get_binding_index('output')

        # Get binding shapes
        input_shape = self.engine.get_binding_shape(self.input_binding)
        output_shape = self.engine.get_binding_shape(self.output_binding)

        # Allocate host and device buffers
        self.host_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
        self.host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)

        self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)
        self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)

        # Create CUDA stream
        self.stream = cuda.Stream()

    def image_callback(self, msg):
        """Process image with TensorRT optimized YOLO"""
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image for TensorRT
            input_image = self.preprocess_image(cv_image)

            # Copy input to GPU
            cuda.memcpy_htod_async(self.cuda_input, input_image, self.stream)

            # Execute inference
            self.context.execute_async_v2(
                bindings=[int(self.cuda_input), int(self.cuda_output)],
                stream_handle=self.stream.handle
            )

            # Copy output from GPU
            cuda.memcpy_dtoh_async(self.host_output, self.cuda_output, self.stream)
            self.stream.synchronize()

            # Process outputs
            detections = self.postprocess_output(self.host_output, cv_image.shape)

            # Publish results
            detection_array_msg = self.create_detection_message(detections, msg.header)
            self.detection_pub.publish(detection_array_msg)

            self.get_logger().info(f'Published {len(detections)} detections')

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')

    def preprocess_image(self, image):
        """Preprocess image for TensorRT inference"""
        # Resize image to model input size (640x640 for YOLOv8)
        input_h, input_w = 640, 640
        resized = cv2.resize(image, (input_w, input_h))

        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Transpose from HWC to CHW
        transposed = normalized.transpose(2, 0, 1)

        # Flatten to 1D array
        flattened = transposed.ravel()

        return flattened

    def postprocess_output(self, output, original_shape):
        """Post-process TensorRT output to get detections"""
        # This is a simplified post-processing
        # In practice, you'd need to decode YOLO outputs properly
        # including applying NMS (Non-Maximum Suppression)

        # Reshape output based on YOLO architecture
        output = output.reshape(1, -1, 84)  # 84 = 4 bbox + 1 conf + 80 classes (COCO)

        # Apply confidence threshold
        conf_threshold = 0.5
        detections = []

        for detection in output[0]:
            bbox = detection[:4]  # x1, y1, x2, y2
            conf = detection[4]
            class_probs = detection[5:]

            if conf > conf_threshold:
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]

                if class_score > conf_threshold:
                    detections.append({
                        'bbox': bbox,
                        'confidence': conf,
                        'class_id': class_id,
                        'class_score': class_score
                    })

        return detections

    def create_detection_message(self, detections, header):
        """Create ROS detection message from detections"""
        from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose

        detection_array_msg = Detection2DArray()
        detection_array_msg.header = header

        for det in detections:
            detection_msg = Detection2D()
            detection_msg.header = header

            # Convert bbox coordinates to image space
            # This is a simplified conversion - adjust based on your camera parameters
            bbox = BoundingBox2D()
            bbox.size_x = int(det['bbox'][2] - det['bbox'][0])
            bbox.size_y = int(det['bbox'][3] - det['bbox'][1])
            bbox.center.x = int(det['bbox'][0] + bbox.size_x / 2)
            bbox.center.y = int(det['bbox'][1] + bbox.size_y / 2)

            detection_msg.bbox = bbox

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det['class_id'])
            hypothesis.hypothesis.score = det['confidence']
            detection_msg.results.append(hypothesis)

            detection_array_msg.detections.append(detection_msg)

        return detection_array_msg


def main(args=None):
    rclpy.init(args=args)
    node = TensorRTYOLONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Model Conversion to TensorRT

To convert a YOLO model to TensorRT format:

```python
# convert_to_tensorrt.py
import torch
from ultralytics import YOLO
from torch2trt import torch2trt
import cv2

def convert_yolo_to_tensorrt(model_path, output_path):
    """Convert YOLO model to TensorRT format"""
    # Load YOLO model
    model = YOLO(model_path)

    # Set model to evaluation mode
    model.model.eval()

    # Create dummy input for TensorRT conversion
    dummy_input = torch.randn(1, 3, 640, 640).cuda()

    # Convert to TensorRT
    model_trt = torch2trt(
        model.model,
        [dummy_input],
        fp16_mode=True,
        max_workspace_size=1<<25  # 32MB
    )

    # Save the TensorRT model
    with open(output_path, 'wb') as f:
        f.write(model_trt.engine.serialize())

    print(f"TensorRT model saved to {output_path}")

if __name__ == "__main__":
    convert_yolo_to_tensorrt('yolov8n.pt', 'yolov8n.engine')
```

## Part 3: Integration with Robotic Actions

### Action Triggering Node

Now let's create a node that uses detection results to trigger robotic actions:

```python
#!/usr/bin/env python3
"""
Object Detection Action Trigger Node
This node subscribes to detection results and triggers robotic actions
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math


class DetectionActionNode(Node):
    def __init__(self):
        super().__init__('detection_action_node')

        # Subscribe to detection results
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        # Publisher for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publisher for status messages
        self.status_pub = self.create_publisher(String, '/robot/status', 10)

        # Define target objects
        self.target_objects = ['person', 'cup', 'bottle', 'chair']

        self.get_logger().info('Detection Action Node initialized')

    def detection_callback(self, msg):
        """Process detection results and trigger actions"""
        target_found = False
        closest_object = None
        min_distance = float('inf')

        for detection in msg.detections:
            # Get the most confident detection of a target object
            for result in detection.results:
                if result.hypothesis.class_id in self.target_objects:
                    if result.hypothesis.score > 0.7:  # Confidence threshold
                        # Calculate distance from center (simplified)
                        image_center_x = 320  # Assuming 640x480 image
                        distance_from_center = abs(detection.bbox.center.x - image_center_x)

                        if distance_from_center < min_distance:
                            min_distance = distance_from_center
                            closest_object = detection
                            target_found = True

        if target_found and closest_object:
            # Generate action based on detection
            action = self.generate_action(closest_object)

            # Publish action
            self.cmd_vel_pub.publish(action)

            # Publish status
            status_msg = String()
            status_msg.data = f"Tracking {closest_object.results[0].hypothesis.class_id}"
            self.status_pub.publish(status_msg)

    def generate_action(self, detection):
        """Generate action based on detection"""
        cmd_vel = Twist()

        # Calculate offset from image center
        image_center_x = 320  # Assuming 640x480 image
        offset_x = detection.bbox.center.x - image_center_x

        # Proportional control for rotation
        rotation_kp = 0.005
        cmd_vel.angular.z = -offset_x * rotation_kp  # Negative for correct direction

        # Move forward if object is centered enough
        center_threshold = 50  # pixels
        if abs(offset_x) < center_threshold:
            cmd_vel.linear.x = 0.2  # Move forward
        else:
            cmd_vel.linear.x = 0.0  # Don't move forward when turning

        return cmd_vel


def main(args=None):
    rclpy.init(args=args)
    node = DetectionActionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 4: Evaluation and Performance Metrics

### Performance Evaluation Node

Create a node to evaluate the performance of the detection system:

```python
#!/usr/bin/env python3
"""
Object Detection Performance Evaluation Node
This node evaluates detection performance metrics
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import collections


class DetectionEvaluationNode(Node):
    def __init__(self):
        super().__init__('detection_evaluation_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to detection results
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        # Performance tracking
        self.detection_times = collections.deque(maxlen=100)
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()

        # Setup timer for periodic reporting
        self.timer = self.create_timer(5.0, self.report_performance)

        self.get_logger().info('Detection Evaluation Node initialized')

    def detection_callback(self, msg):
        """Process detection results for performance evaluation"""
        current_time = time.time()
        self.detection_times.append(current_time)
        self.detection_count += len(msg.detections)
        self.frame_count += 1

    def report_performance(self):
        """Report performance metrics"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if self.frame_count > 0:
            avg_fps = self.frame_count / elapsed_time
            avg_detection_time = 0
            if len(self.detection_times) > 1:
                time_diffs = [self.detection_times[i] - self.detection_times[i-1]
                             for i in range(1, len(self.detection_times))]
                if time_diffs:
                    avg_detection_time = sum(time_diffs) / len(time_diffs)

            avg_detections_per_frame = self.detection_count / self.frame_count if self.frame_count > 0 else 0

            self.get_logger().info(f'Performance Report:')
            self.get_logger().info(f'  Average FPS: {avg_fps:.2f}')
            self.get_logger().info(f'  Average Detection Time: {avg_detection_time:.4f}s ({1/avg_detection_time:.2f}Hz if applicable)')
            self.get_logger().info(f'  Average Detections per Frame: {avg_detections_per_frame:.2f}')
            self.get_logger().info(f'  Total Frames Processed: {self.frame_count}')
            self.get_logger().info(f'  Total Detections: {self.detection_count}')
            self.get_logger().info(f'  Elapsed Time: {elapsed_time:.2f}s')


def main(args=None):
    rclpy.init(args=args)
    node = DetectionEvaluationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 5: Practical Exercise

### Exercise: Object Following Robot

Create a complete application that makes a robot follow a specific object:

```python
#!/usr/bin/env python3
"""
Object Following Robot
Complete application that makes a robot follow a detected object
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class ObjectFollowingRobot(Node):
    def __init__(self):
        super().__init__('object_following_robot')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to camera images and detections
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        # Subscribe to laser scan for obstacle avoidance
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Publisher for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Robot state
        self.current_detections = []
        self.obstacle_distance = float('inf')
        self.target_object = 'person'  # Object to follow
        self.following = False

        self.get_logger().info('Object Following Robot initialized')

    def image_callback(self, msg):
        """Process camera images"""
        # This callback can be used for additional image processing
        pass

    def detection_callback(self, msg):
        """Process detection results"""
        self.current_detections = []

        for detection in msg.detections:
            for result in detection.results:
                if result.hypothesis.class_id == self.target_object and result.hypothesis.score > 0.7:
                    self.current_detections.append(detection)

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Get distance at forward direction (index around the middle)
        if len(msg.ranges) > 0:
            forward_idx = len(msg.ranges) // 2
            self.obstacle_distance = msg.ranges[forward_idx]

    def run(self):
        """Main control loop"""
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.control_loop)

    def control_loop(self):
        """Main control loop for object following"""
        cmd_vel = Twist()

        if self.current_detections:
            # Get the largest detection (closest object)
            largest_detection = max(
                self.current_detections,
                key=lambda d: d.bbox.size_x * d.bbox.size_y
            )

            # Calculate offset from image center
            image_center_x = 320  # Assuming 640x480 image
            offset_x = largest_detection.bbox.center.x - image_center_x

            # Proportional control for rotation
            rotation_kp = 0.005
            cmd_vel.angular.z = -offset_x * rotation_kp

            # Move forward if object is centered and no obstacles
            center_threshold = 80  # pixels
            if (abs(offset_x) < center_threshold and
                self.obstacle_distance > 0.8):  # 0.8m safety distance
                cmd_vel.linear.x = 0.3  # Move forward
                self.following = True
            else:
                cmd_vel.linear.x = 0.0
        else:
            # Stop if no target detected
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.following = False

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Log status
        if self.following:
            self.get_logger().info('Following target object')
        else:
            self.get_logger().info('No target detected')


def main(args=None):
    rclpy.init(args=args)
    robot = ObjectFollowingRobot()
    robot.run()
    rclpy.spin(robot)
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Lab Assessment

### Assessment Tasks

Complete the following tasks to demonstrate your understanding:

1. **Basic Implementation**:
   - Implement the basic YOLO detection node
   - Verify that it publishes detection messages
   - Test with a sample image topic

2. **Performance Optimization**:
   - Convert the YOLO model to TensorRT format
   - Compare inference times between original and optimized models
   - Document the performance improvement

3. **Integration**:
   - Connect the detection node with the action triggering node
   - Test that the robot responds to detected objects
   - Evaluate the response time and accuracy

4. **Evaluation**:
   - Run the performance evaluation node
   - Measure FPS, detection accuracy, and latency
   - Document your findings

### Questions for Reflection

1. How does TensorRT optimization affect the accuracy of YOLO detections?
2. What are the trade-offs between detection accuracy and inference speed?
3. How would you modify the system to detect custom objects specific to your robotic application?
4. What safety considerations should be taken into account when using object detection for robotic control?

## Summary

In this lab, you've implemented a complete YOLO-based object detection system for robotics applications. You learned how to:
- Set up and configure YOLO models for robotic applications
- Integrate YOLO with ROS 2 for real-time object detection
- Optimize performance using NVIDIA TensorRT
- Use detection results to trigger robotic actions
- Evaluate system performance

The object detection system you've built forms a crucial component of Vision-Language-Action systems, providing the visual perception capabilities needed for robots to understand and interact with their environment. The integration with robotic actions demonstrates how computer vision can be used to enable autonomous behaviors in robotic systems.