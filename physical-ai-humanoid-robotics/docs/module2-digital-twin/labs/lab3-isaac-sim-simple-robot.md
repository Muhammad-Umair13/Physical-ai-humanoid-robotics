---
sidebar_label: 'Lab 3: Isaac Sim Simple Robot Simulation'
sidebar_position: 3
---

# Lab 3: Isaac Sim Simple Robot Simulation

## Objective

In this lab, you will create a simple robot simulation in NVIDIA Isaac Sim, learn to configure physics properties, implement basic control systems, and integrate with ROS 2. You will understand how to leverage Isaac Sim's high-fidelity physics and rendering for humanoid robotics development.

## Prerequisites

- NVIDIA Isaac Sim installed (Omniverse environment)
- Python 3.8+ environment
- Basic knowledge of USD (Universal Scene Description)
- Understanding of ROS 2 concepts (optional but helpful)

## Learning Outcomes

By the end of this lab, you will be able to:
1. Set up and configure Isaac Sim for robot simulation
2. Create a simple robot using USD and Isaac Sim APIs
3. Configure physics properties and materials
4. Implement basic robot control systems
5. Integrate with ROS 2 for communication
6. Generate synthetic sensor data for AI training

## Step 1: Setting Up Isaac Sim Environment

First, let's create a Python script to initialize Isaac Sim and set up the basic environment. Create a file called `simple_robot_sim.py`:

```python
import omni
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import set_stage_units
from omni.isaac.core.utils import viewports
from pxr import Gf, Sdf, UsdGeom
import numpy as np
import carb

def setup_isaac_sim():
    """Initialize Isaac Sim and configure basic settings"""

    # Set stage units to meters
    set_stage_units(1.0)  # 1 unit = 1 meter

    # Get world instance
    world = World(stage_units_in_meters=1.0)

    # Add ground plane
    world.scene.add_default_ground_plane()

    return world

def configure_physics():
    """Configure physics settings for the simulation"""
    # Get physics scene settings
    physics_settings = carb.settings.get_settings()

    # Set physics parameters
    physics_settings.set("/physics/solverPositionIterationCount", 8)
    physics_settings.set("/physics/solverVelocityIterationCount", 1)
    physics_settings.set("/physics/timeStepsPerSecond", 60)  # 60 Hz physics update
    physics_settings.set("/physics/maxSubSteps", 1)

    # Set gravity
    UsdGeom.SetStageMetersPerUnit(omni.usd.get_context().get_stage(), 1.0)

def create_simple_robot(world):
    """Create a simple wheeled robot in Isaac Sim"""
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims import RigidPrim, XFormPrim
    from omni.isaac.core.utils.prims import create_primitive
    from omni.isaac.core.utils.stage import add_reference_to_stage

    # Create robot root
    robot_root_path = "/World/MyRobot"
    robot_xform = XFormPrim(prim_path=robot_root_path, name="my_robot")

    # Create robot body (chassis)
    chassis_path = f"{robot_root_path}/Chassis"
    create_primitive(
        prim_path=chassis_path,
        primitive_type="Cuboid",
        position=np.array([0, 0, 0.2]),
        orientation=np.array([0, 0, 0, 1]),
        scale=np.array([0.5, 0.3, 0.15]),
        color=np.array([0.1, 0.1, 0.8])
    )

    # Create wheels
    wheel_radius = 0.1
    wheel_width = 0.05
    wheel_positions = [
        [0.2, 0.2, 0.1],   # front right
        [0.2, -0.2, 0.1],  # front left
        [-0.2, 0.2, 0.1],  # back right
        [-0.2, -0.2, 0.1]  # back left
    ]

    for i, pos in enumerate(wheel_positions):
        wheel_path = f"{robot_root_path}/Wheel_{i}"
        create_primitive(
            prim_path=wheel_path,
            primitive_type="Cylinder",
            position=pos,
            orientation=np.array([0, 0, 0, 1]),
            scale=np.array([wheel_radius, wheel_width, wheel_radius]),
            color=np.array([0.2, 0.2, 0.2])
        )

    # Add rigid body properties to chassis
    from omni.physx.scripts import physicsUtils

    stage = omni.usd.get_context().get_stage()
    chassis_prim = stage.GetPrimAtPath(chassis_path)

    # Set mass properties
    physicsUtils.setMass(chassis_prim, mass=2.0)

    return robot_root_path

def main():
    """Main function to run the simulation"""
    print("Setting up Isaac Sim environment...")

    # Initialize Isaac Sim
    world = setup_isaac_sim()
    configure_physics()

    # Create simple robot
    robot_path = create_simple_robot(world)

    # Reset the world to apply all changes
    world.reset()

    print("Robot created successfully!")
    print("Robot path:", robot_path)

    # Run simulation for a few steps to see the robot
    for i in range(100):
        world.step(render=True)
        if i % 20 == 0:
            print(f"Simulation step {i}")

    print("Simulation completed.")

if __name__ == "__main__":
    main()
```

## Step 2: Creating a More Complex Humanoid Robot

Now let's create a more complex humanoid robot. Create a file called `humanoid_robot.py`:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import set_stage_units
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils import nucleus
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.semantics import add_update_semantics

class HumanoidRobot(Articulation):
    def __init__(
        self,
        prim_path: str,
        name: str = "humanoid_robot",
        usd_path: str = None,
        position: np.ndarray = np.array([0, 0, 1.0]),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
    ) -> None:
        """Initialize a simple humanoid robot"""

        self._stage = omni.usd.get_context().get_stage()

        # Create the root prim
        self.create_humanoid_robot(prim_path, position, orientation)

        super().__init__(
            prim_path=prim_path,
            name=name,
        )

    def create_humanoid_robot(self, prim_path, position, orientation):
        """Create the humanoid robot structure"""

        # Create root body (pelvis)
        pelvis_path = f"{prim_path}/pelvis"
        self.create_body_segment(
            pelvis_path,
            position=position,
            size=[0.2, 0.25, 0.15],
            mass=5.0
        )

        # Create torso
        torso_path = f"{prim_path}/torso"
        self.create_body_segment(
            torso_path,
            position=[position[0], position[1], position[2] + 0.25],
            size=[0.2, 0.3, 0.15],
            mass=10.0
        )

        # Create head
        head_path = f"{prim_path}/head"
        self.create_body_segment(
            head_path,
            position=[position[0], position[1], position[2] + 0.65],
            size=[0.2, 0.2, 0.2],
            mass=2.0,
            shape="Sphere"
        )

        # Create left leg
        self.create_leg(f"{prim_path}/left_leg", position, "left")

        # Create right leg
        self.create_leg(f"{prim_path}/right_leg", position, "right")

        # Create left arm
        self.create_arm(f"{prim_path}/left_arm", position, "left")

        # Create right arm
        self.create_arm(f"{prim_path}/right_arm", position, "right")

        # Create joints to connect body parts
        self.create_joints(prim_path)

    def create_body_segment(self, prim_path, position, size, mass, shape="Box"):
        """Create a body segment (box or sphere)"""
        from omni.isaac.core.utils.prims import create_primitive

        if shape == "Box":
            create_primitive(
                prim_path=prim_path,
                primitive_type="Cuboid",
                position=position,
                scale=size,
                color=np.array([0.1, 0.1, 0.8]) if "torso" in prim_path else
                       np.array([0.8, 0.2, 0.2]) if "leg" in prim_path else
                       np.array([0.8, 0.8, 0.8])
            )
        elif shape == "Sphere":
            create_primitive(
                prim_path=prim_path,
                primitive_type="Sphere",
                position=position,
                scale=size,
                color=np.array([0.8, 0.8, 0.8])
            )

        # Add rigid body properties
        self.add_rigid_body(prim_path, mass)

    def create_leg(self, leg_root_path, base_position, side):
        """Create a leg with hip, knee, and ankle joints"""
        offset_x = 0.1 if side == "right" else -0.1

        # Upper leg (thigh)
        thigh_path = f"{leg_root_path}/upper_leg"
        self.create_body_segment(
            thigh_path,
            position=[base_position[0] + offset_x, base_position[1], base_position[2] - 0.1],
            size=[0.08, 0.08, 0.3],
            mass=3.0
        )

        # Lower leg (shin)
        shin_path = f"{leg_root_path}/lower_leg"
        self.create_body_segment(
            shin_path,
            position=[base_position[0] + offset_x, base_position[1], base_position[2] - 0.4],
            size=[0.07, 0.07, 0.3],
            mass=2.5
        )

        # Foot
        foot_path = f"{leg_root_path}/foot"
        self.create_body_segment(
            foot_path,
            position=[base_position[0] + offset_x, base_position[1], base_position[2] - 0.65],
            size=[0.15, 0.08, 0.06],
            mass=1.0
        )

    def create_arm(self, arm_root_path, base_position, side):
        """Create an arm with shoulder, elbow joints"""
        offset_x = 0.25 if side == "right" else -0.25
        offset_y = 0.3 if side == "right" else -0.3

        # Upper arm
        upper_arm_path = f"{arm_root_path}/upper_arm"
        self.create_body_segment(
            upper_arm_path,
            position=[base_position[0] + offset_x, base_position[1] + offset_y, base_position[2] + 0.4],
            size=[0.06, 0.06, 0.25],
            mass=1.5
        )

        # Lower arm
        lower_arm_path = f"{arm_root_path}/lower_arm"
        self.create_body_segment(
            lower_arm_path,
            position=[base_position[0] + offset_x, base_position[1] + offset_y, base_position[2] + 0.15],
            size=[0.05, 0.05, 0.25],
            mass=1.0
        )

    def add_rigid_body(self, prim_path, mass):
        """Add rigid body properties to a prim"""
        from omni.physx.scripts import physicsUtils

        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            # Set mass
            physicsUtils.setMass(prim, mass)

            # Set other physics properties
            UsdPhysics.MassAPI.Apply(prim)

    def create_joints(self, robot_path):
        """Create joints to connect body parts"""
        # This is a simplified version - in a real implementation you'd create actual joints
        # For now, we'll just ensure the parts are properly connected in the scene graph
        pass

def setup_humanoid_simulation():
    """Set up the Isaac Sim environment with a humanoid robot"""
    # Set stage units
    set_stage_units(1.0)

    # Get world instance
    world = World(stage_units_in_meters=1.0)

    # Add ground plane
    world.scene.add_default_ground_plane()

    # Create humanoid robot
    robot = HumanoidRobot(
        prim_path="/World/MyHumanoid",
        name="my_humanoid"
    )

    # Add robot to world
    world.scene.add(robot)

    return world, robot

def main():
    """Main function to run the humanoid simulation"""
    print("Setting up Isaac Sim with humanoid robot...")

    # Initialize simulation
    world, robot = setup_humanoid_simulation()

    # Reset the world
    world.reset()

    print("Humanoid robot created successfully!")

    # Run simulation for a few steps
    for i in range(200):
        world.step(render=True)

        if i % 50 == 0:
            print(f"Simulation step {i}")

            # Print robot state
            joint_positions = robot.get_joints_state().positions
            print(f"Joint positions: {len(joint_positions)} joints")

    print("Simulation completed.")

if __name__ == "__main__":
    main()
```

## Step 3: Adding Physics and Control

Create a more advanced script with physics and control called `humanoid_control.py`:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import set_stage_units
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.semantics import add_update_semantics

class HumanoidController:
    def __init__(self, world, robot):
        self.world = world
        self.robot = robot
        self.joint_names = [
            "left_hip_joint", "left_knee_joint", "left_ankle_joint",
            "right_hip_joint", "right_knee_joint", "right_ankle_joint",
            "left_shoulder_joint", "left_elbow_joint",
            "right_shoulder_joint", "right_elbow_joint"
        ]

        # Initialize joint controllers
        self.setup_joint_control()

    def setup_joint_control(self):
        """Setup joint control parameters"""
        # For this example, we'll use simple position control
        # In a real implementation, you would configure PID controllers
        pass

    def move_to_position(self, joint_positions):
        """Move robot joints to specified positions"""
        # Set joint positions
        self.robot.set_joints_state(positions=np.array(joint_positions))

    def get_robot_state(self):
        """Get current robot state"""
        joint_state = self.robot.get_joints_state()
        return {
            'positions': joint_state.positions,
            'velocities': joint_state.velocities,
            'efforts': joint_state efforts if available else None
        }

    def simple_walk_cycle(self, phase):
        """Generate simple walking pattern based on phase"""
        # Simplified walking gait - in reality this would be much more complex
        positions = []

        # Left leg (swing phase when right is stance)
        positions.append(np.sin(phase) * 0.3)     # left_hip
        positions.append(np.sin(phase + 0.5) * 0.5)  # left_knee
        positions.append(np.sin(phase - 0.2) * 0.1)  # left_ankle

        # Right leg (stance phase when left is swing)
        positions.append(np.sin(phase + np.pi) * 0.3)  # right_hip
        positions.append(np.sin(phase + np.pi + 0.5) * 0.5)  # right_knee
        positions.append(np.sin(phase + np.pi - 0.2) * 0.1)  # right_ankle

        # Arms (counterbalance)
        positions.append(np.sin(phase + np.pi) * 0.2)  # left_shoulder
        positions.append(np.sin(phase) * 0.1)          # left_elbow
        positions.append(np.sin(phase) * 0.2)          # right_shoulder
        positions.append(np.sin(phase + np.pi) * 0.1)  # right_elbow

        return positions

def setup_advanced_simulation():
    """Set up Isaac Sim with advanced features"""
    # Set stage units
    set_stage_units(1.0)

    # Get world instance
    world = World(stage_units_in_meters=1.0)

    # Add ground plane with texture
    world.scene.add_default_ground_plane(
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.1
    )

    # Set up camera view
    set_camera_view(eye=[2, 2, 1.5], target=[0, 0, 0.8])

    # Create a simple humanoid (using a pre-built asset for this example)
    # In a real scenario, you might load a more complex robot
    asset_root_path = nucleus.get_assets_root_path()
    if asset_root_path is None:
        print("Could not find Isaac Sim assets. Using simple cube robot.")

        # Create a simple robot using basic shapes
        robot_path = "/World/SimpleRobot"
        add_reference_to_stage(
            usd_path="realsense_camera.usd",  # This is just an example
            prim_path=robot_path
        )
    else:
        # Try to load a basic robot if available
        robot_path = "/World/Robot"
        try:
            add_reference_to_stage(
                usd_path=asset_root_path + "/Isaac/Robots/Franka/franka_instanceable.usd",
                prim_path=robot_path
            )
        except:
            print("Could not load Franka robot, creating simple robot instead")
            # Create simple robot using basic shapes
            create_simple_robot_shapes(robot_path)

    # Add sensors
    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([1.5, 1.5, 1.0]),
        frequency=30,
        resolution=(640, 480)
    )
    world.scene.add(camera)

    return world

def create_simple_robot_shapes(robot_path):
    """Create a simple robot using basic shapes"""
    from omni.isaac.core.utils.prims import create_primitive

    # Create base body
    create_primitive(
        prim_path=robot_path + "/base",
        primitive_type="Cuboid",
        position=np.array([0, 0, 0.5]),
        scale=np.array([0.3, 0.3, 0.3]),
        color=np.array([0.1, 0.1, 0.8])
    )

def main():
    """Main function to run the advanced simulation"""
    print("Setting up advanced Isaac Sim environment...")

    # Initialize simulation
    world = setup_advanced_simulation()

    # Reset the world
    world.reset()

    print("Advanced simulation setup complete!")

    # Initialize controller
    controller = HumanoidController(world, None)  # We'll use a different approach for the robot

    # Run simulation with control loop
    phase = 0
    for i in range(600):  # Run for 10 seconds at 60 FPS
        world.step(render=True)

        # Update walking pattern every few steps
        if i % 3 == 0:
            phase += 0.1
            target_positions = controller.simple_walk_cycle(phase)
            # In a real implementation, you would apply these positions to the robot

        if i % 60 == 0:  # Print every second
            print(f"Simulation time: {i/60:.1f}s")

    print("Advanced simulation completed.")

if __name__ == "__main__":
    main()
```

## Step 4: Creating a ROS 2 Bridge Example

Create a ROS 2 bridge script called `isaac_sim_ros_bridge.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Time
import numpy as np
import threading
import time

class IsaacSimROSBridge(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros_bridge')

        # Publishers for sensor data from Isaac Sim
        self.joint_state_publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        self.imu_publisher = self.create_publisher(
            Imu,
            '/imu/data',
            10
        )

        # Subscribers for commands to Isaac Sim
        self.joint_cmd_subscriber = self.create_subscription(
            Float64MultiArray,
            '/joint_group_position_controller/command',
            self.joint_cmd_callback,
            10
        )

        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Timer for publishing sensor data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

        # Robot state
        self.current_joint_positions = [0.0] * 10
        self.current_joint_velocities = [0.0] * 10
        self.robot_pose = [0.0, 0.0, 0.0]  # x, y, theta

        # Lock for thread safety
        self.state_lock = threading.Lock()

        self.get_logger().info('Isaac Sim ROS Bridge initialized')

    def joint_cmd_callback(self, msg):
        """Handle joint position commands from ROS"""
        with self.state_lock:
            if len(msg.data) == len(self.current_joint_positions):
                self.current_joint_positions = list(msg.data)
                self.get_logger().debug(f'Received joint commands: {self.current_joint_positions}')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS"""
        # In a real implementation, this would control the robot in Isaac Sim
        self.get_logger().info(f'Received velocity command: linear={msg.linear}, angular={msg.angular}')

        # Update robot pose based on velocity (simplified)
        dt = 0.1  # Time step
        with self.state_lock:
            self.robot_pose[0] += msg.linear.x * dt
            self.robot_pose[1] += msg.linear.y * dt if hasattr(msg.linear, 'y') else 0.0
            self.robot_pose[2] += msg.angular.z * dt

    def publish_sensor_data(self):
        """Publish sensor data to ROS"""
        # Publish joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = 'base_link'
        joint_state_msg.name = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        with self.state_lock:
            joint_state_msg.position = self.current_joint_positions.copy()
            joint_state_msg.velocity = self.current_joint_velocities.copy()
            joint_state_msg.effort = [0.0] * len(self.current_joint_positions)

        self.joint_state_publisher.publish(joint_state_msg)

        # Publish IMU data (simulated)
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Simulated orientation (level for now)
        imu_msg.orientation.w = 1.0
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0

        # Simulated angular velocity
        imu_msg.angular_velocity = Vector3(x=0.0, y=0.0, z=0.0)

        # Simulated linear acceleration (gravity + movement)
        imu_msg.linear_acceleration = Vector3(x=0.0, y=0.0, z=9.81)

        self.imu_publisher.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)

    bridge = IsaacSimROSBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 5: Creating a Complete Simulation Example

Create a comprehensive simulation script called `complete_humanoid_sim.py`:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import set_stage_units
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.materials import PhysicsMaterial
import numpy as np
import asyncio
import carb

class CompleteHumanoidSim:
    def __init__(self):
        self.world = None
        self.robot = None
        self.camera = None
        self.physics_material = None

    def setup_environment(self):
        """Set up the complete simulation environment"""
        # Set stage units to meters
        set_stage_units(1.0)

        # Get world instance
        self.world = World(stage_units_in_meters=1.0)

        # Configure physics
        self.configure_physics()

        # Add ground plane
        self.world.scene.add_default_ground_plane(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.1
        )

        # Set up camera
        self.setup_camera()

        # Load robot (using a pre-built asset if available)
        self.load_robot()

        # Create physics materials
        self.create_physics_materials()

        # Set camera view
        set_camera_view(eye=[2.5, 2.5, 1.5], target=[0, 0, 0.8])

    def configure_physics(self):
        """Configure physics settings"""
        physics_settings = carb.settings.get_settings()
        physics_settings.set("/physics/solverPositionIterationCount", 8)
        physics_settings.set("/physics/solverVelocityIterationCount", 1)
        physics_settings.set("/physics/timeStepsPerSecond", 120)  # Higher frequency for better stability
        physics_settings.set("/physics/maxSubSteps", 2)

    def setup_camera(self):
        """Set up camera for the simulation"""
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.array([2.5, 2.5, 1.5]),
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.camera)

    def load_robot(self):
        """Load robot asset"""
        asset_root_path = get_assets_root_path()

        if asset_root_path is not None:
            # Try to load a humanoid robot if available in the assets
            robot_path = "/World/Robot"
            try:
                add_reference_to_stage(
                    usd_path=asset_root_path + "/Isaac/Robots/Humanoid/humanoid_instanceable.usd",
                    prim_path=robot_path
                )
                print("Loaded humanoid robot from assets")
            except:
                print("Could not load humanoid robot from assets, using simple setup")
                # For this example, we'll proceed with a basic setup
        else:
            print("Could not find Isaac Sim assets, using basic setup")

    def create_physics_materials(self):
        """Create physics materials for different surfaces"""
        self.physics_material = PhysicsMaterial(
            prim_path="/World/Looks/robot_material",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.1
        )

    def run_simulation(self, duration=10.0):
        """Run the simulation for a specified duration"""
        # Reset the world
        self.world.reset()

        print(f"Starting simulation for {duration} seconds...")

        # Calculate number of steps (assuming 60 FPS rendering)
        steps = int(duration * 60)

        for i in range(steps):
            # Step the physics simulation
            self.world.step(render=True)

            # Occasionally print progress
            if i % 600 == 0:  # Every 10 seconds of sim time
                elapsed_time = i / 60.0
                print(f"Simulation time: {elapsed_time:.1f}s")

            # Example: Move robot periodically (this would be controlled by external commands in real use)
            if i % 60 == 0:  # Every second
                self.execute_behavior()

        print("Simulation completed.")

    def execute_behavior(self):
        """Execute a simple behavior (e.g., move joints)"""
        # This is a placeholder for more complex behaviors
        # In a real implementation, this would interface with control algorithms
        pass

def main():
    """Main function to run the complete simulation"""
    print("Setting up complete Isaac Sim humanoid environment...")

    sim = CompleteHumanoidSim()
    sim.setup_environment()
    sim.run_simulation(duration=10.0)

if __name__ == "__main__":
    main()
```

## Step 6: Running the Simulation

To run these simulations in Isaac Sim:

1. **Start Isaac Sim** using the Omniverse App Launcher or directly

2. **Open the Isaac Sim application**

3. **Run the Python scripts** from within Isaac Sim's scripting environment:
   - Go to Window → Script Editor
   - Copy and paste the code into the editor
   - Run the script

4. **Alternatively**, run from command line if Isaac Sim Python API is available:
   ```bash
   python complete_humanoid_sim.py
   ```

## Step 7: Testing and Validation

### Basic Functionality Test
1. Verify that the robot model loads correctly
2. Check that physics simulation is stable
3. Confirm that sensors are publishing data
4. Test basic movement and control

### Performance Testing
1. Monitor simulation frame rate
2. Check physics stability
3. Validate sensor data quality
4. Test with different robot configurations

### Integration Testing
1. Verify ROS 2 bridge functionality
2. Test sensor data transmission
3. Validate control command execution
4. Check system stability over extended runs

## Expected Results

After completing this lab, you should have:
- A working Isaac Sim environment with a humanoid robot
- Understanding of USD and Isaac Sim APIs
- Basic control system implementation
- ROS 2 integration for communication
- Experience with sensor simulation and synthetic data generation

## Troubleshooting

### Common Issues and Solutions

1. **Physics Instability**: Increase solver iterations or reduce time step
2. **Slow Performance**: Simplify models or reduce physics complexity
3. **Asset Loading Errors**: Verify Isaac Sim installation and asset paths
4. **ROS Connection Issues**: Check network configuration and bridge setup

### Performance Optimization Tips

1. Use simplified collision geometry where possible
2. Adjust physics parameters for your specific use case
3. Limit the number of active sensors during development
4. Use level-of-detail approaches for complex scenes

## Extensions

Try these extensions to enhance your understanding:

1. Implement more sophisticated control algorithms (PID, MPC)
2. Add computer vision capabilities with synthetic data generation
3. Create reinforcement learning environments
4. Implement more realistic humanoid behaviors
5. Add multiple robots to the simulation
6. Integrate with perception and planning systems

## Summary

In this lab, you've learned to create and control robots in NVIDIA Isaac Sim, leveraging its high-fidelity physics and rendering capabilities. You've explored USD scene description, implemented basic control systems, and integrated with ROS 2 for communication. Isaac Sim's capabilities for synthetic data generation and AI training make it a powerful tool for humanoid robotics development.