---
sidebar_label: 'ROS 2 Architecture'
sidebar_position: 2
---

# ROS 2 Architecture

## Overview

The architecture of ROS 2 is fundamentally different from ROS 1, primarily due to its underlying communication middleware. ROS 2 uses DDS (Data Distribution Service) as its communication layer, which provides a standardized middleware for real-time, distributed, and fault-tolerant applications.

## DDS (Data Distribution Service) Layer

DDS is the foundation of ROS 2's communication system. It provides:

- **Publisher/Subscriber Model**: Nodes publish data to topics and subscribe to topics to receive data
- **Request/Reply Model**: Services use this pattern for synchronous communication
- **Discovery**: Automatic discovery of participants in the network
- **Quality of Service (QoS) Policies**: Configurable policies for reliability, durability, and other communication characteristics

### Quality of Service (QoS) Profiles

QoS profiles allow fine-tuning of communication behavior:

- **Reliability**: Best effort or reliable delivery
- **Durability**: Volatile or transient local data persistence
- **History**: Keep all or only the last N samples
- **Deadline**: Maximum time between sample publications
- **Lifespan**: Maximum lifetime of samples

## Client Library (rclcpp/rclpy)

The client libraries provide the interface between user code and the DDS implementation:

- **rclcpp**: C++ client library
- **rclpy**: Python client library

These libraries handle the mapping between ROS 2 concepts (nodes, topics, services) and DDS entities.

## Node Architecture

### Node Structure

Each node in ROS 2 contains:

- **Node Handle**: Interface to the ROS 2 system
- **Executors**: Manage the execution of callbacks
- **Parameters**: Configurable values that can be changed at runtime
- **Timers**: Periodic callback execution
- **Guard Conditions**: Event-based callback execution

### Executors

Executors determine how callbacks are processed:

- **Single-threaded Executor**: All callbacks run in a single thread
- **Multi-threaded Executor**: Callbacks run in multiple threads
- **Static Single-threaded Executor**: Optimized single-threaded executor

## Communication Patterns

### Publish/Subscribe

The publish/subscribe pattern enables asynchronous communication:

```python
# Publisher example
publisher = node.create_publisher(String, 'topic_name', 10)

# Subscriber example
subscriber = node.create_subscription(String, 'topic_name', callback, 10)
```

### Services

Services provide synchronous request/reply communication:

```python
# Service server
service = node.create_service(AddTwoInts, 'add_two_ints', callback)

# Service client
client = node.create_client(AddTwoInts, 'add_two_ints')
```

### Actions

Actions are designed for long-running tasks:

```python
# Action server
action_server = ActionServer(node, Fibonacci, 'fibonacci', execute_callback)

# Action client
action_client = ActionClient(node, Fibonacci, 'fibonacci')
```

## Parameter System

ROS 2 includes a dynamic parameter system:

- Parameters can be declared with types, descriptions, and constraints
- Parameters can be changed at runtime
- Parameter callbacks can react to parameter changes
- Parameters can be loaded from YAML files

## Launch System

The launch system in ROS 2 provides:

- **Launch Files**: XML, YAML, or Python files to start multiple nodes
- **Composition**: Running multiple nodes in the same process
- **Lifecycle Nodes**: Nodes with explicit state management
- **Conditional Launch**: Starting nodes based on conditions

## Security Architecture

ROS 2 includes security features:

- **Authentication**: Identity verification
- **Access Control**: Authorization of entities
- **Encryption**: Data encryption in transit
- **Signing**: Message authentication

## Real-time Considerations

ROS 2 supports real-time applications:

- **Lock-free data structures**: Reduce contention
- **Deadline policies**: Ensure timing requirements
- **Memory allocation control**: Predictable allocation patterns
- **Thread priorities**: Configurable thread priorities

## Module Integration for Humanoid Robotics

In humanoid robotics, the architecture supports:

- **Sensor Fusion**: Multiple sensor nodes publishing to common topics
- **Control Hierarchy**: Joint controllers, balance controllers, and high-level planners
- **Safety Systems**: Emergency stop mechanisms and safety monitors
- **Simulation Integration**: Seamless transition between simulation and real hardware

## Summary

The ROS 2 architecture provides a robust, scalable foundation for complex robotic systems. Its use of DDS enables distributed computing, real-time capabilities, and multi-platform support. The modular design with QoS profiles allows for fine-tuning communication behavior to meet specific requirements of humanoid robotics applications.