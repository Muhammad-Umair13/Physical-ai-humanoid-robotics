---
sidebar_label: 'Nodes, Topics, Services, Actions'
sidebar_position: 3
---

# Nodes, Topics, Services, and Actions

## Overview

In ROS 2, communication between different parts of a robotic system happens through a set of well-defined patterns. Understanding these patterns is crucial for developing effective humanoid robotics applications. This chapter covers the four primary communication patterns: nodes, topics, services, and actions.

## Nodes

Nodes are the fundamental building blocks of ROS 2 applications. Each node performs a specific task and communicates with other nodes through various communication mechanisms.

### Creating a Node

```cpp
// C++ example
#include "rclcpp/rclcpp.hpp"

class MinimalNode : public rclcpp::Node
{
public:
  MinimalNode() : Node("minimal_node") {
    RCLCPP_INFO(this->get_logger(), "Hello from minimal node!");
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalNode>());
  rclcpp::shutdown();
  return 0;
}
```

```python
# Python example
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from minimal node!')

def main(args=None):
    rclpy.init(args=args)
    minimal_node = MinimalNode()
    rclpy.spin(minimal_node)
    minimal_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Parameters

Nodes can be configured with parameters that can be changed at runtime:

```python
# Declaring and using parameters
self.declare_parameter('robot_name', 'humanoid_robot')
robot_name = self.get_parameter('robot_name').value
```

## Topics and Publishers/Subscribers

Topics enable asynchronous communication through a publish/subscribe model. Publishers send messages to topics, and subscribers receive messages from topics.

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Quality of Service (QoS) in Publishers and Subscribers

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a QoS profile for reliable communication
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Use QoS profile when creating publisher/subscriber
publisher = self.create_publisher(String, 'topic', qos_profile)
subscriber = self.create_subscription(String, 'topic', callback, qos_profile)
```

## Services

Services provide synchronous request/reply communication. A client sends a request to a service, and the service processes the request and sends back a response.

### Service Server Example

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Example

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info('Result of add_two_ints: %d' % response.sum)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

Actions are designed for long-running tasks that may take a significant amount of time to complete. They provide feedback during execution and can be canceled.

### Action Server Example

```python
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            self.get_logger().info('Publishing feedback: {0}'.format(
                feedback_msg.sequence))

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Returning result: {0}'.format(result.sequence))

        return result
```

### Action Client Example

```python
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(
            feedback.sequence))

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))
```

## Communication Patterns in Humanoid Robotics

### Sensor Data Distribution

Humanoid robots have numerous sensors (cameras, IMUs, force sensors, joint encoders) that continuously publish data:

- Joint states published to `/joint_states`
- IMU data published to `/imu/data`
- Camera images published to `/camera/image_raw`
- Force/torque data published to `/ft_sensor/wrench`

### Control Commands

Control commands are typically sent via services or actions:

- Walking pattern generation via actions
- Joint position commands via topics
- Emergency stop via services
- Behavior switching via services

### Coordination Between Subsystems

Different subsystems coordinate through various communication patterns:

- Perception nodes publish object detections
- Planning nodes subscribe to sensor data and publish motion plans
- Control nodes execute motion plans and publish feedback
- Monitoring nodes aggregate system status

## Best Practices

### Topic Design
- Use descriptive names that follow ROS naming conventions
- Consider QoS settings based on the nature of the data
- Use appropriate message types or define custom ones when needed

### Service Design
- Use services for operations that have a clear request/response pattern
- Consider timeout settings for service calls
- Handle service unavailability gracefully

### Action Design
- Use actions for long-running operations that need feedback
- Design appropriate feedback messages to keep clients informed
- Implement proper cancellation handling

## Summary

Understanding the different communication patterns in ROS 2 is essential for building robust humanoid robotics applications. Each pattern serves a specific purpose and choosing the right one for your use case is crucial for system performance and reliability. Topics are ideal for continuous data streams, services for discrete operations, and actions for long-running tasks with feedback.