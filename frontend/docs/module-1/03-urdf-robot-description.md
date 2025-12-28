---
sidebar_position: 3
---

# URDF Robot Description

The Unified Robot Description Format (URDF) is the standard XML-based format for describing robot geometry, kinematics, and visual properties in ROS. This chapter provides comprehensive coverage of URDF—from basic link and joint definitions through advanced features like Xacro macros and Gazebo integration. Understanding URDF is essential for any ROS-based robotics project, as it provides the foundation for visualization, simulation, and motion planning.

## URDF Fundamentals

URDF represents robots as trees of rigid bodies (links) connected by joints. This tree structure mirrors the physical construction of robots while enabling efficient computation of kinematics and dynamics.

### Basic URDF Structure

A complete URDF file consists of several key elements:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
    <!-- Material definitions for colors -->
    <material name="metal">
        <color rgba="0.5 0.5 0.5 1.0"/>
    </material>

    <!-- Link definitions: rigid bodies of the robot -->
    <link name="base_link">
        <visual>
            <geometry>
                <box size="0.3 0.3 0.15"/>
            </geometry>
            <material name="metal"/>
        </visual>
        <collision>
            <geometry>
                <box size="0.3 0.3 0.15"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="5.0"/>
            <inertia ixx="0.1" ixy="0" ixz="0"
                     iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
    </link>

    <!-- Joint definitions: connections between links -->
    <joint name="waist_joint" type="revolute">
        <parent link="base_link"/>
        <child link="torso_link"/>
        <origin xyz="0 0 0.15"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    </joint>

    <link name="torso_link">
        <!-- Link definition continues... -->
    </link>
</robot>
```

This structure follows several important principles:

**Tree Structure**: The robot must form a tree—no loops or multiple paths between links. This ensures unique kinematic chains and prevents ambiguity in forward kinematics.

**Parent-Child Relationships**: Each joint has exactly one parent link and one child link. The child link moves relative to the parent link through the joint transformation.

**Coordinate Conventions**: ROS uses a right-handed coordinate system where:
- X points forward (for typical robot orientations)
- Y points left
- Z points up

### Link Elements

A link represents a rigid body in the robot. Each link can have three types of descriptions:

**Visual Elements** define how the link appears in RViz and other visualization tools:

```xml
<link name="upper_arm">
    <visual>
        <!-- Geometry defines the shape -->
        <geometry>
            <!-- Box: width, height, depth -->
            <box size="0.05 0.05 0.3"/>
        </geometry>
        <!-- Origin defines local coordinate frame -->
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
        <!-- Material defines color and texture -->
        <material name="aluminum">
            <color rgba="0.8 0.8 0.8 1.0"/>
        </material>
    </visual>
</link>
```

Multiple visual elements can describe a single link—this is common when a robot has separate parts that should be visualized differently:

```xml
<link name="hand">
    <visual>
        <geometry>
            <box size="0.08 0.02 0.1"/>
        </geometry>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </visual>
    <!-- Palm -->
    <visual>
        <geometry>
            <box size="0.06 0.02 0.08"/>
        </geometry>
        <origin xyz="0 0 0.04" rpy="0 0 0"/>
    </visual>
    <!-- Fingers represented as small boxes -->
    <visual>
        <geometry>
            <box size="0.01 0.01 0.04"/>
        </geometry>
        <origin xyz="-0.02 0 0.08" rpy="0 0 0"/>
    </visual>
    <visual>
        <geometry>
            <box size="0.01 0.01 0.04"/>
        </geometry>
        <origin xyz="0 0 0.08" rpy="0 0 0"/>
    </visual>
    <visual>
        <geometry>
            <box size="0.01 0.01 0.04"/>
        </geometry>
        <origin xyz="0.02 0 0.08" rpy="0 0 0"/>
    </visual>
</link>
```

**Collision Elements** define the geometry used for collision detection:

```xml
<link name="torso">
    <!-- Visual geometry: detailed mesh for appearance -->
    <visual>
        <geometry>
            <mesh filename="package://robot_description/meshes/torso.stl"/>
        </geometry>
    </visual>

    <!-- Collision geometry: simplified for performance -->
    <collision>
        <geometry>
            <box size="0.25 0.2 0.35"/>
        </geometry>
        <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
</link>
```

The separation between visual and collision geometry is crucial. Visual geometry can be complex meshes for appearance, while collision geometry should be simplified shapes (boxes, cylinders, spheres) for efficient computation.

**Inertial Elements** define mass properties for physics simulation:

```xml
<link name="upper_leg">
    <inertial>
        <!-- Mass in kilograms -->
        <mass value="3.5"/>

        <!-- Origin of the center of mass in the link frame -->
        <origin xyz="0 0 0.15"/>

        <!-- Inertia tensor: Ixx, Ixy, Ixz, Iyy, Iyz, Izz -->
        <!-- For a uniform cylinder along Z: -->
        <inertia ixx="0.02" ixy="0" ixz="0"
                 iyy="0.02" iyz="0" izz="0.01"/>
    </inertial>
</link>
```

Computing accurate inertia tensors is essential for realistic simulation. For simple shapes:

**Sphere**: Ixx = Iyy = Izz = (2/5) × m × r²

**Cylinder along axis**: Ixx = Iyy = (1/4) × m × r² + (1/12) × m × h², Izz = (1/2) × m × r²

**Box**: Ixx = (1/12) × m × (h² + d²), Iyy = (1/12) × m × (w² + d²), Izz = (1/12) × m × (w² + h²)

### Joint Types

Joints define how links move relative to each other. URDF supports several joint types:

**Fixed Joint**: No motion allowed. Used for permanent connections:

```xml
<joint name="camera_mount" type="fixed">
    <parent link="head_link"/>
    <child link="camera_link"/>
    <origin xyz="0.1 0 0.05" rpy="0 0 0"/>
</joint>
```

**Revolute Joint**: Rotates around a single axis within limits:

```xml
<joint name="elbow_joint" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="forearm_link"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/> <!-- Rotate around Y axis -->
    <!-- Limits are required for revolute joints -->
    <limit lower="-2.5" upper="2.5"
           effort="50" velocity="5.0"/>
</joint>
```

**Continuous Joint**: Rotates indefinitely around an axis:

```xml
<joint name="neck_yaw" type="continuous">
    <parent link="torso_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.4"/>
    <axis xyz="0 0 1"/> <!-- Rotate around Z axis -->
    <!-- No limits for continuous joints -->
</joint>
```

**Prismatic Joint**: Linear motion along an axis:

```xml
<joint name="linear_actuator" type="prismatic">
    <parent link="base_link"/>
    <child link="slider_link"/>
    <origin xyz="0.1 0 0"/>
    <axis xyz="0 0 1"/> <!-- Linear motion along Z -->
    <limit lower="0" upper="0.3"
           effort="200" velocity="0.1"/>
</joint>
```

**Floating Joint**: Six degrees of freedom without constraints. Used when no simpler joint type suffices:

```xml
<joint name="floating_base" type="floating">
    <parent link="world"/>
    <child link="base_link"/>
    <!-- No origin or axis elements for floating joints -->
</joint>
```

**Planar Joint**: Motion in a plane (2 translation + 1 rotation):

```xml
<joint name="planar_joint" type="planar">
    <parent link="ground"/>
    <child link="mobile_base"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/> <!-- Normal to the plane of motion -->
</joint>
```

## Creating a Complete Humanoid URDF

Building a complete humanoid requires careful organization. Here's a simplified example demonstrating proper structure:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
    <!-- ==================== MATERIALS ==================== -->
    <material name="white">
        <color rgba="1.0 1.0 1.0 1.0"/>
    </material>
    <material name="dark_gray">
        <color rgba="0.3 0.3 0.3 1.0"/>
    </material>
    <material name="skin">
        <color rgba="0.96 0.80 0.69 1.0"/>
    </material>

    <!-- ==================== BASE ==================== -->
    <link name="pelvis">
        <visual>
            <geometry>
                <box size="0.2 0.15 0.1"/>
            </geometry>
            <material name="dark_gray"/>
            <origin xyz="0 0 0"/>
        </visual>
        <collision>
            <geometry>
                <box size="0.2 0.15 0.1"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="2.0"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="0.01" ixy="0" ixz="0"
                     iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
    </link>

    <!-- ==================== LEGS ==================== -->
    <!-- Right Hip Joint -->
    <joint name="right_hip_yaw" type="revolute">
        <parent link="pelvis"/>
        <child link="right_hip_link"/>
        <origin xyz="0.08 0 -0.05"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.0" upper="1.0" effort="100" velocity="2.0"/>
    </joint>

    <link name="right_hip_link">
        <visual>
            <geometry>
                <cylinder radius="0.06" height="0.08"/>
            </geometry>
            <material name="dark_gray"/>
            <origin xyz="0 0 0.04"/>
        </visual>
        <inertial>
            <mass value="1.0"/>
            <origin xyz="0 0 0.04"/>
            <inertia ixx="0.005" ixy="0" ixz="0"
                     iyy="0.005" iyz="0" izz="0.002"/>
        </inertial>
    </link>

    <!-- Right Knee Joint -->
    <joint name="right_knee" type="revolute">
        <parent link="right_hip_link"/>
        <child link="right_thigh_link"/>
        <origin xyz="0 0 0.08"/>
        <axis xyz="1 0 0"/>
        <limit lower="-2.0" upper="0" effort="80" velocity="3.0"/>
    </joint>

    <link name="right_thigh_link">
        <visual>
            <geometry>
                <cylinder radius="0.045" height="0.25"/>
            </geometry>
            <material name="white"/>
            <origin xyz="0 0 0.125"/>
        </visual>
        <collision>
            <geometry>
                <cylinder radius="0.045" height="0.25"/>
            </geometry>
            <origin xyz="0 0 0.125"/>
        </collision>
        <inertial>
            <mass value="2.5"/>
            <origin xyz="0 0 0.125"/>
            <inertia ixx="0.02" ixy="0" ixz="0"
                     iyy="0.02" iyz="0" izz="0.005"/>
        </inertial>
    </link>

    <!-- Continue with ankle joint and foot... -->

    <!-- ==================== SPINE ==================== -->
    <joint name="waist_yaw" type="revolute">
        <parent link="pelvis"/>
        <child link="torso_link"/>
        <origin xyz="0 0 0.05"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.5" upper="1.5" effort="150" velocity="1.5"/>
    </joint>

    <link name="torso_link">
        <visual>
            <geometry>
                <box size="0.22 0.15 0.35"/>
            </geometry>
            <material name="white"/>
            <origin xyz="0 0 0.175"/>
        </visual>
        <inertial>
            <mass value="5.0"/>
            <origin xyz="0 0 0.175"/>
            <inertia ixx="0.08" ixy="0" ixz="0"
                     iyy="0.08" iyz="0" izz="0.04"/>
        </inertial>
    </link>

    <!-- ==================== ARMS ==================== -->
    <!-- Left Shoulder Joint -->
    <joint name="left_shoulder_yaw" type="revolute">
        <parent link="torso_link"/>
        <child link="left_shoulder_link"/>
        <origin xyz="0.15 0 0.35"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.5" upper="1.5" effort="50" velocity="3.0"/>
    </joint>

    <!-- Continue with shoulder pitch, elbow, wrist... -->

    <!-- ==================== HEAD ==================== -->
    <joint name="neck_yaw" type="revolute">
        <parent link="torso_link"/>
        <child link="neck_link"/>
        <origin xyz="0 0 0.4"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.0" upper="1.0" effort="10" velocity="2.0"/>
    </joint>

    <link name="neck_link">
        <visual>
            <cylinder radius="0.04" height="0.08"/>
            <material name="skin"/>
            <origin xyz="0 0 0.04"/>
        </visual>
        <inertial>
            <mass value="0.5"/>
            <origin xyz="0 0 0.04"/>
            <inertia ixx="0.001" ixy="0" ixz="0"
                     iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
    </link>

    <joint name="head_joint" type="fixed">
        <parent link="neck_link"/>
        <child link="head_link"/>
        <origin xyz="0 0 0.08"/>
    </joint>

    <link name="head_link">
        <visual>
            <sphere radius="0.08"/>
            <material name="skin"/>
        </visual>
        <!-- Eyes -->
        <visual>
            <sphere radius="0.015"/>
            <material name="black"/>
            <origin xyz="0.04 0.03 0.02"/>
        </visual>
        <visual>
            <sphere radius="0.015"/>
            <material name="black"/>
            <origin xyz="0.04 -0.03 0.02"/>
        </visual>
        <inertial>
            <mass value="1.0"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="0.002" ixy="0" ixz="0"
                     iyy="0.002" iyz="0" izz="0.002"/>
        </inertial>
    </link>
</robot>
```

## Advanced URDF Features

### Xacro: URDF Macro Language

Xacro (XML Macro) enables parameterized URDF files, reducing duplication and enabling configuration:

```xml
<?xml version="1.0"?>
<robot name="humanoid_xacro" xmlns:xacro="http://www.ros.org/wiki/xacro">
    <!-- Define properties -->
    <xacro:property name="body_color" value="0.3 0.3 0.3 1.0"/>
    <xacro:property name="skin_color" value="0.96 0.80 0.69 1.0"/>

    <!-- Define a reusable link macro -->
    <xacro:macro name="cylinder_link" params="name mass radius height color">
        <link name="${name}">
            <visual>
                <geometry>
                    <cylinder radius="${radius}" height="${height}"/>
                </geometry>
                <material name="${color}"/>
                <origin xyz="0 0 ${height/2}"/>
            </visual>
            <collision>
                <geometry>
                    <cylinder radius="${radius}" height="${height}"/>
                </geometry>
                <origin xyz="0 0 ${height/2}"/>
            </collision>
            <inertial>
                <mass value="${mass}"/>
                <origin xyz="0 0 ${height/2}"/>
                <inertia ixx="${mass * (3*radius*radius + height*height)/12}"
                         ixy="0" ixz="0"
                         iyy="${mass * (3*radius*radius + height*height)/12}"
                         iyz="0" izz="${mass * radius * radius / 2}"/>
            </inertial>
        </link>
    </xacro:macro>

    <!-- Define a joint macro -->
    <xacro:macro name="revolute_joint" params="name parent child axis xyz limit_upper limit_lower effort velocity">
        <joint name="${name}" type="revolute">
            <parent link="${parent}"/>
            <child link="${child}"/>
            <origin xyz="${xyz}"/>
            <axis xyz="${axis}"/>
            <limit lower="${limit_lower}" upper="${limit_upper}"
                   effort="${effort}" velocity="${velocity}"/>
        </joint>
    </xacro:macro>

    <!-- Use macros to build the robot -->
    <xacro:cylinder_link name="torso"
                         mass="5.0"
                         radius="0.12"
                         height="0.35"
                         color="body_color"/>

    <xacro:cylinder_link name="right_thigh"
                         mass="2.5"
                         radius="0.045"
                         height="0.25"
                         color="body_color"/>

    <xacro:revolute_joint name="right_hip"
                          parent="torso"
                          child="right_thigh"
                          axis="1 0 0"
                          xyz="0.08 0 -0.17"
                          limit_upper="1.5"
                          limit_lower="-2.0"
                          effort="100"
                          velocity="3.0"/>
</robot>
```

Xacro files must be processed before use:

```bash
# Convert xacro to urdf
xacro robot.xacro -o robot.urdf

# Or use directly with ros2 launch
ros2 launch robot_description robot.launch.py
```

### Transmission Elements

Transmissions connect joints to actuators, essential for hardware integration:

```xml
<transmission name="right_hip_transmission">
    <type>transmission_interface/SimpleTransmission</type>

    <joint name="right_hip_yaw">
        <hardwareInterface>PositionJointInterface</hardwareInterface>
    </joint>

    <actuator name="right_hip_motor">
        <hardwareInterface>PositionJointInterface</hardwareInterface>
        <mechanicalReduction>100</mechanicalReduction>
    </actuator>
</transmission>
```

### Gazebo Simulation Elements

Gazebo-specific tags enable physics simulation:

```xml
<gazebo reference="right_thigh_link">
    <!-- Material properties for friction -->
    <mu1>0.9</mu1>
    <mu2>0.9</mu2>

    <!-- Contact coefficients -->
    <kp>1000000.0</kp>
    <kd>100.0</kd>

    <!-- Self-collision -->
    <selfCollide>true</selfCollide>
</gazebo>

<gazebo reference="right_hip_joint">
    <physics>
        <ode>
            <implicitSpringDamper>true</implicitSpringDamper>
            <cfm_damping>0.2</cfm_damping>
        </ode>
    </physics>
</gazebo>
```

## Validation and Tools

### URDF Validation

Always validate your URDF before use:

```bash
# Check URDF syntax
check_urdf robot.urdf

# Visualize the kinematic tree
urdf_to_graphiz robot.urdf
```

### RViz Visualization

RViz provides 3D visualization of URDF models:

```bash
# Install robot_state_publisher
sudo apt install ros-humble-robot-state-publisher

# Launch with RViz
ros2 launch robot_description display.launch.py
```

The display.launch.py file typically contains:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    robot_description_path = FindPackageShare('robot_description')

    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': open(
                    robot_description_path.joinpath('urdf/robot.urdf')
                ).read()
            }]
        ),

        # Joint state publisher (for manual joint control)
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', robot_description_path.joinpath('rviz/display.rviz')],
        ),
    ])
```

### Common URDF Errors

**Disconnected Links**: All links must be connected through joint chains. Use `check_urdf` to detect disconnected components.

**Incomplete Inertia**: Every link needs an inertial element for simulation. Missing inertials cause physics errors.

**Invalid Origin**: Origin coordinates must be within reasonable bounds. Check for swapped XYZ or RPY values.

**Missing Limits**: Revolute and prismatic joints require limit elements. Without limits, some tools cannot process the URDF.

## Key Takeaways

URDF provides the foundation for representing robot geometry in ROS. Understanding these concepts is essential:

- **Tree Structure**: Robots form trees with unique parent-child chains
- **Links and Joints**: Links are rigid bodies; joints define their connections and motion types
- **Visual, Collision, Inertial**: Each link needs all three for complete specification
- **Xacro Macros**: Reduce duplication and enable configuration
- **Validation**: Always validate URDF before use

With robot geometry defined, we can now explore how to launch and manage complex ROS 2 systems—the topic of the next chapter.
