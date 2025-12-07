---
sidebar_label: 'Creating a Gazebo World'
sidebar_position: 3
---

# Creating a Gazebo World

## Overview

Creating a Gazebo world involves designing an environment where robots can operate, interact, and be tested. A well-designed world includes appropriate terrain, obstacles, lighting, and physics parameters that match the intended application. This chapter covers the process of creating realistic and functional Gazebo worlds for humanoid robotics applications.

## World File Structure

Gazebo worlds are defined using the Simulation Description Format (SDF). The basic structure includes:

```xml
<sdf version='1.7'>
  <world name='my_world'>
    <!-- Environment elements -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics type='ode'>
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Models and objects -->
    <model name='my_robot'>
      <!-- Robot definition -->
    </model>

    <!-- Lighting -->
    <light name='sun_light' type='directional'>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>
  </world>
</sdf>
```

## Essential World Components

### Ground Plane
The ground plane provides a basic surface for robots to operate on:

```xml
<include>
  <uri>model://ground_plane</uri>
</include>
<state world_name="default">
  <model name="ground_plane">
    <pose>0 0 0 0 0 0</pose>
  </model>
</state>
```

### Lighting System
Proper lighting is crucial for camera sensors and visual appeal:

```xml
<light name='sun' type='directional'>
  <cast_shadows>true</cast_shadows>
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-0.3 0.3 -1</direction>
</light>
```

### Physics Configuration
Configure physics parameters for realistic simulation:

```xml
<physics type='ode' name='default_physics' default='0'>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
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
```

## Creating Custom Environments

### Indoor Environments
For humanoid robots operating indoors, consider:

```xml
<!-- Indoor environment with walls and furniture -->
<model name='wall_1'>
  <static>true</static>
  <link name='link'>
    <collision name='collision'>
      <geometry>
        <box><size>10 0.2 3</size></box>
      </geometry>
    </collision>
    <visual name='visual'>
      <geometry>
        <box><size>10 0.2 3</size></box>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Doorways and passages -->
<model name='doorway'>
  <static>true</static>
  <pose>5 0 0 0 0 0</pose>
  <link name='left_jamb'>
    <collision name='collision'>
      <geometry>
        <box><size>0.2 0.1 2</size></box>
      </geometry>
    </collision>
    <visual name='visual'>
      <geometry>
        <box><size>0.2 0.1 2</size></box>
      </geometry>
    </visual>
  </link>
  <link name='right_jamb'>
    <pose>1.8 0 0 0 0 0</pose>
    <collision name='collision'>
      <geometry>
        <box><size>0.2 0.1 2</size></box>
      </geometry>
    </collision>
    <visual name='visual'>
      <geometry>
        <box><size>0.2 0.1 2</size></box>
      </geometry>
    </visual>
  </link>
</model>
```

### Outdoor Environments
For outdoor humanoid robot testing:

```xml
<!-- Terrain with varying elevation -->
<model name='terrain'>
  <static>true</static>
  <link name='link'>
    <collision name='collision'>
      <geometry>
        <heightmap>
          <uri>model://terrain/images/heightmap.png</uri>
          <size>100 100 10</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name='visual'>
      <geometry>
        <heightmap>
          <uri>model://terrain/images/heightmap.png</uri>
          <size>100 100 10</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>

<!-- Obstacles and navigation challenges -->
<model name='obstacle_1'>
  <pose>10 5 0.5 0 0 0</pose>
  <link name='link'>
    <collision name='collision'>
      <geometry>
        <cylinder>
          <radius>0.5</radius>
          <length>1.0</length>
        </cylinder>
      </geometry>
    </collision>
    <visual name='visual'>
      <geometry>
        <cylinder>
          <radius>0.5</radius>
          <length>1.0</length>
        </cylinder>
      </geometry>
    </visual>
  </link>
</model>
```

## Adding Humanoid Robot Models

### Spawning a Humanoid Robot
Example of including a humanoid robot in the world:

```xml
<model name='simple_humanoid'>
  <pose>0 0 1 0 0 0</pose>
  <link name='torso'>
    <pose>0 0 0.5 0 0 0</pose>
    <collision name='collision'>
      <geometry>
        <box><size>0.3 0.3 0.5</size></box>
      </geometry>
    </collision>
    <visual name='visual'>
      <geometry>
        <box><size>0.3 0.3 0.5</size></box>
      </geometry>
    </visual>
    <inertial>
      <mass>10.0</mass>
      <inertia>
        <ixx>0.5</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>0.5</iyy>
        <iyz>0</iyz>
        <izz>0.5</izz>
      </inertia>
    </inertial>
  </link>

  <!-- Head -->
  <link name='head'>
    <pose>0 0 0.75 0 0 0</pose>
    <collision name='collision'>
      <geometry>
        <sphere><radius>0.15</radius></sphere>
      </geometry>
    </collision>
    <visual name='visual'>
      <geometry>
        <sphere><radius>0.15</radius></sphere>
      </geometry>
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

  <!-- Connect torso and head -->
  <joint name='neck_joint' type='revolute'>
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
</model>
```

## Advanced World Features

### Weather Simulation
Add atmospheric effects:

```xml
<scene>
  <shadows>true</shadows>
  <sky>
    <clouds>
      <speed>0.6</speed>
      <direction>0.8 0.1</direction>
      <humidity>0.5</humidity>
      <mean_size>0.5</mean_size>
      <ambient>0.8 0.8 0.8 1</ambient>
    </clouds>
  </sky>
</scene>
```

### Sensor Calibration Areas
Designate areas for sensor testing:

```xml
<model name='calibration_target'>
  <pose>15 0 1 0 0 0</pose>
  <link name='link'>
    <visual name='visual'>
      <geometry>
        <box><size>1 1 0.01</size></box>
      </geometry>
      <material>
        <script>
          <uri>file://media/materials/scripts/gazebo.material</uri>
          <name>Gazebo/CalibrationTarget</name>
        </script>
      </material>
    </visual>
  </link>
</model>
```

## World Testing and Validation

### Physics Validation
Test the world with various scenarios:
- Robot walking on different terrains
- Object manipulation tasks
- Collision detection accuracy
- Stability under various conditions

### Performance Testing
- Monitor simulation speed
- Check for physics instabilities
- Validate sensor outputs
- Ensure consistent behavior

## World Optimization

### Performance Considerations
- Use simplified collision geometry where possible
- Limit the number of complex models
- Adjust physics parameters for performance
- Use static models for non-moving objects

### Memory Management
- Keep model complexity reasonable
- Use instancing for repeated objects
- Optimize textures and materials
- Consider level-of-detail approaches

## Best Practices for Humanoid Robotics

### Environment Design
- Create scenarios that match intended use cases
- Include obstacles and challenges relevant to humanoid capabilities
- Design for both indoor and outdoor testing
- Include navigation and manipulation challenges

### Safety Considerations
- Design safe testing areas for robot validation
- Include recovery mechanisms for failed behaviors
- Plan for robot falls and emergency stops
- Consider robot-human interaction scenarios

### Realism vs. Performance
- Balance visual fidelity with simulation speed
- Use appropriate physics parameters for humanoid dynamics
- Include realistic sensor noise and limitations
- Validate simulation results against real-world data

## Example Complete World File

Here's a complete example of a humanoid testing world:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <physics type="ode">
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

    <!-- Indoor testing area -->
    <model name='testing_room'>
      <static>true</static>
      <link name='floor'>
        <collision name='collision'>
          <geometry>
            <box><size>10 10 0.1</size></box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box><size>10 10 0.1</size></box>
          </geometry>
        </visual>
      </link>
      <model name='wall_north'>
        <pose>0 5 1.5 0 0 0</pose>
        <link name='link'>
          <collision name='collision'>
            <geometry>
              <box><size>10 0.2 3</size></box>
            </geometry>
          </collision>
          <visual name='visual'>
            <geometry>
              <box><size>10 0.2 3</size></box>
            </geometry>
          </visual>
        </link>
      </model>
      <!-- Additional walls and obstacles would go here -->
    </model>

    <!-- Lighting -->
    <light name='main_light' type='point'>
      <pose>0 0 5 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>20</range>
        <constant>0.5</constant>
        <linear>0.1</linear>
        <quadratic>0.01</quadratic>
      </attenuation>
    </light>
  </world>
</sdf>
```

## Summary

Creating effective Gazebo worlds for humanoid robotics requires careful consideration of physics parameters, environment design, and performance optimization. A well-designed world enables comprehensive testing of humanoid robot capabilities while maintaining simulation stability and performance. The key is to balance realism with computational efficiency to create environments that accurately reflect real-world challenges while remaining practical for development and testing.