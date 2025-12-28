---
sidebar_position: 2
---

# World Building with SDF

Simulation Description Format (SDF) is the XML-based language used by Gazebo and related simulation platforms to describe entire simulation environments. Unlike URDF, which describes individual robots, SDF describes complete worlds including physics, lighting, models, and actors. This chapter provides comprehensive coverage of SDF world building for robotics simulation.

## SDF Architecture and Structure

SDF organizes simulation environments hierarchically, with worlds containing models, which in turn contain links and joints. This structure mirrors physical reality while enabling efficient simulation.

### World File Structure

A complete SDF world file begins with the root element and includes all simulation components:

```xml
<?xml version="1.0" ?>
<sdf version="1.11">
    <world name="humanoid_lab">
        <!-- Physics settings -->
        <physics name="default_physics" default="true" type="ode">
            <gravity>0 0 -9.8</gravity>
            <ode>
                <solver>
                    <type>quick</type>
                    <iters>100</iters>
                    <sor>1.0</sor>
                </solver>
                <constraints>
                    <cfm>0.0</cfm>
                    <erp>0.2</erp>
                    <contact_max_correcting_vel>100</contact_max_correcting_vel>
                    <contact_surface_layer>0.001</contact_surface_layer>
                </constraints>
            </ode>
        </physics>

        <!-- Scene definition -->
        <scene>
            <ambient>0.4 0.4 0.4 1.0</ambient>
            <background>0.7 0.7 0.7 1.0</background>
            <shadows>true</shadows>
            <grid>true</grid>
        </scene>

        <!-- Lighting -->
        <light name="sun" type="directional">
            <pose>0 0 10 0 0 0</pose>
            <diffuse>0.8 0.8 0.8 1.0</diffuse>
            <specular>0.2 0.2 0.2 1.0</specular>
            <cast_shadows>true</cast_shadows>
            <intensity>1.0</intensity>
            <direction>-0.5 -1 -0.5</direction>
        </light>

        <!-- Ground plane -->
        <model name="ground">
            <static>true</static>
            <link name="ground_link">
                <collision name="collision">
                    <geometry>
                        <plane>
                            <normal>0 0 1</normal>
                            <size>100 100</size>
                        </plane>
                    </geometry>
                    <surface>
                        <friction>
                            <ode>
                                <mu>0.8</mu>
                                <mu2>0.8</mu2>
                            </ode>
                        </friction>
                    </surface>
                </collision>
                <visual name="visual">
                    <geometry>
                        <plane>
                            <normal>0 0 1</normal>
                            <size>100 100</size>
                        </plane>
                    </geometry>
                    <material>
                        <script>
                            <uri>file://media/materials/scripts/gazebo.material</uri>
                            <name>Gazebo/Grey</name>
                        </script>
                    </material>
                </visual>
            </link>
        </model>

        <!-- Include humanoid robot -->
        <include>
            <uri>model://humanoid_robot</uri>
            <name>humanoid</name>
            <pose>0 0 0 0 0 0</pose>
        </include>

        <!-- Environment models -->
        <include>
            <uri>model://table</uri>
            <name>work_table</name>
            <pose>2.0 0.5 0 0 0 0</pose>
        </include>

        <!-- Population of obstacles -->
        <population name="obstacle_field">
            <model name="box_obstacle">
                <link name="link">
                    <visual name="visual">
                        <geometry>
                            <box size="0.3 0.3 0.5"/>
                        </geometry>
                        <material>
                            <script>
                                <uri>file://media/materials/scripts/gazebo.material</uri>
                                <name>Gazebo/Red</name>
                            </script>
                        </material>
                    </visual>
                </link>
            </model>
            <pose>5 0 0.25 0 0 0</pose>
            <box>
                <size>3 2 0.1</size>
                <cell_count>5</cell_count>
            </box>
        </population>
    </world>
</sdf>
```

This structure demonstrates several key SDF concepts:

**Version Declaration**: The `version` attribute on the root element specifies which SDF version the file uses. Different versions have different features and syntax.

**Nested Hierarchy**: Worlds contain models, models contain links, and links contain geometries, materials, and collision specifications. This mirrors the physical hierarchy of robotic systems.

**Multiple Configurations**: Physics, scene, and lighting are all configured within the world file, giving complete control over the simulation environment.

### Physics Engine Configuration

SDF supports multiple physics engines, each with configurable parameters:

```xml
<!-- ODE (Open Dynamics Engine) - Fast and stable for most cases -->
<physics name="ode_physics" type="ode">
    <gravity>0 0 -9.8</gravity>
    <ode>
        <solver>
            <type>quick</type>  <!-- fast but less accurate -->
            <iters>50</iters>   <!-- more iterations = more accuracy -->
            <sor>1.0</sor>      <!-- successive over-relaxation -->
        </solver>
        <constraints>
            <cfm>0.0</cfm>      <!-- constraint force mixing -->
            <erp>0.2</erp>      <!-- error reduction parameter -->
        </constraints>
    </ode>
</physics>

<!-- DART - More accurate for complex contacts -->
<physics name="dart_physics" type="dart">
    <gravity>0 0 -9.8</gravity>
    <dart>
        <solver>
            <type>ant</type>
            <iters>100</iters>
        </solver>
    </dart>
</physics>

<!-- Bullet - Good balance of speed and accuracy -->
<physics name="bullet_physics" type="bullet">
    <gravity>0 0 -9.8</gravity>
    <bullet>
        <solver>
            <type>pgs</type>
            <iters>50</iters>
        </solver>
    </bullet>
</physics>
```

For humanoid simulation, the choice of physics engine affects contact stability and joint friction modeling. ODE works well for most applications, while DART provides better handling of complex multi-contact scenarios common in bipedal locomotion.

### Scene and Atmospheric Effects

Realistic scene rendering requires proper lighting and atmospheric configuration:

```xml
<scene>
    <!-- Ambient light - base illumination -->
    <ambient>0.3 0.3 0.35 1.0</ambient>

    <!-- Sky color -->
    <background>0.5 0.6 0.8 1.0</background>

    <!-- Fog for depth perception -->
    <fog>
        <color>0.5 0.6 0.8 1.0</color>
        <type>linear</type>
        <start>10</start>
        <end>50</end>
        <density>0.01</density>
    </fog>

    <!-- Shadows enable depth perception -->
    <shadows>true</shadows>

    <!-- Grid helps with orientation -->
    <grid>true</grid>

    <!--skybox for realistic sky rendering -->
    <sky>
        <time>12</time>
        <sunrise>6</sunrise>
        <sunset>18</sunset>
        <clouds>
            <speed>0.1</speed>
            <density>0.2</density>
        </clouds>
    </sky>
</scene>
```

## Model Integration and Spawning

SDF worlds integrate models from multiple sources, including the online model database, local model directories, and inline definitions.

### Including External Models

The `<include>` element loads models from URIs:

```xml
<!-- From online model database -->
<include>
    <uri>model://cafe_table</uri>
    <name>cafe_table_1</name>
    <pose>1.5 0.8 0 0 0 0</pose>
</include>

<!-- From local model directory -->
<include>
    <uri>model://custom_robot</uri>
    <name>my_robot</name>
    <pose>0 0 0 0 0 1.57</pose>
    <static>false</static>
</include>

<!-- From explicit file path -->
<include>
    <uri>file:///home/user/robots/humanoid.urdf</uri>
    <name>humanoid</name>
</include>
```

### Model Composition with Nested Models

Complex environments use nested models for hierarchical organization:

```xml
<model name="lab_environment">
    <!-- Static lab structure -->
    <static>true</static>

    <!-- Floor -->
    <link name="floor">
        <visual name="floor_visual">
            <geometry>
                <box size="10 8 0.1"/>
            </geometry>
            <material>
                <script>
                    <uri>file://materials/scripts/floor.material</uri>
                    <name>LabFloor</name>
                </script>
            </material>
        </visual>
        <collision name="floor_collision">
            <geometry>
                <box size="10 8 0.1"/>
            </geometry>
        </collision>
    </link>

    <!-- Workbench -->
    <link name="workbench_top">
        <pose>3 2 0.8 0 0 0</pose>
        <visual name="bench_top">
            <geometry>
                <box size="2 0.8 0.05"/>
            </geometry>
            <material>
                <script>
                    <uri>file://materials/scripts/bench.material</uri>
                    <name>StainlessSteel</name>
                </script>
            </material>
        </visual>
        <collision name="bench_collision">
            <geometry>
                <box size="2 0.8 0.05"/>
            </geometry>
        </collision>
    </link>

    <!-- Shelf unit -->
    <link name="shelf_1">
        <pose>4 -2 0.4 0 0 0</pose>
        <visual name="shelf_visual">
            <geometry>
                <box size="1.5 0.5 0.02"/>
            </geometry>
        </visual>
    </link>
    <link name="shelf_2">
        <pose>4 -2 0.8 0 0 0</pose>
        <visual name="shelf_visual">
            <geometry>
                <box size="1.5 0.5 0.02"/>
            </geometry>
        </visual>
    </link>
    <link name="shelf_3">
        <pose>4 -2 1.2 0 0 0</pose>
        <visual name="shelf_visual">
            <geometry>
                <box size="1.5 0.5 0.02"/>
            </geometry>
        </visual>
    </link>

    <!-- Objects on shelf -->
    <model name="box_on_shelf">
        <pose>4 -2 1.32 0 0 0</pose>
        <link name="box">
            <visual name="box_visual">
                <geometry>
                    <box size="0.15 0.15 0.15"/>
                </geometry>
                <material>
                    <script>
                        <uri>file://materials/scripts/box.material</uri>
                        <name>Cardboard</name>
                    </script>
                </material>
            </visual>
        </link>
    </model>
</model>
```

### Dynamic Objects with Joints

Objects with moving parts use joints within models:

```xml
<model name="hinged_door">
    <pose>2 1.5 0 0 0 0</pose>
    <link name="door_frame">
        <static>true</static>
        <visual name="frame_visual">
            <geometry>
                <box size="0.1 1.0 2.2"/>
            </geometry>
        </visual>
    </link>

    <link name="door_leaf">
        <pose>0.05 0 1.1 0 0 0</pose>
        <inertial>
            <mass>20</mass>
            <inertia>
                <ixx>10</ixx>
                <iyy>1</iyy>
                <izz>10</izz>
            </inertia>
        </inertial>
        <visual name="door_visual">
            <geometry>
                <box size="0.05 0.9 2.1"/>
            </geometry>
        </visual>
        <collision name="door_collision">
            <geometry>
                <box size="0.05 0.9 2.1"/>
            </geometry>
        </collision>
    </link>

    <joint name="hinge_joint" type="revolute">
        <parent>door_frame</parent>
        <child>door_leaf</child>
        <pose>0 0.45 1.0 0 0 0</pose>
        <axis>
            <xyz>0 0 1</xyz>
        </axis>
        <physics>
            <ode>
                <cfm>0.1</cfm>
                <erp>0.9</erp>
            </ode>
        </physics>
        <limit>
            <lower>-1.57</lower>
            <upper>0</upper>
            <effort>10</effort>
            <velocity>0.5</velocity>
        </limit>
    </joint>
</model>
```

## Lighting Systems

Realistic lighting enables proper camera simulation and visual navigation testing.

### Light Types and Configuration

```xml
<!-- Directional light (sun) -->
<light name="sun" type="directional">
    <pose>10 -10 20 0 0 0</pose>
    <diffuse>0.9 0.9 0.85 1.0</diffuse>
    <specular>0.5 0.5 0.5 1.0</specular>
    <cast_shadows>true</cast_shadows>
    <intensity>1.2</intensity>
    <direction>-0.5 -0.5 -1</direction>
    <shadow>
        <map_size>2048</map_size>
        <near>0.1</near>
        <far>50</far>
    </shadow>
</light>

<!-- Point light (lamp) -->
<light name="desk_lamp" type="point">
    <pose>1.5 0.5 1.8 0 0 0</pose>
    <diffuse>1.0 0.95 0.9 1.0</diffuse>
    <specular>1.0 1.0 1.0 1.0</specular>
    <cast_shadows>false</cast_shadows>
    <intensity>0.8</intensity>
    <range>5</range>
    <decay>2</decay>
</light>

<!-- Spot light (focused beam) -->
<light name="spotlight" type="spot">
    <pose>3 0 3 0 -1.57 0</pose>
    <diffuse>1.0 1.0 1.0 1.0</diffuse>
    <specular>1.0 1.0 1.0 1.0</specular>
    <cast_shadows>true</cast_shadows>
    <intensity>2.0</intensity>
    <direction>0 -1 0</direction>
    <spot>
        <inner_angle>0.3</inner_angle>
        <outer_angle>0.5</outer_angle>
        <falloff>1.0</falloff>
    </spot>
</light>
```

## Advanced SDF Features

### Plugins for Custom Behavior

SDF plugins extend simulation capabilities with custom code:

```xml
<plugin name="contact_sensor_plugin" filename="libcontact_sensor.so">
    <topic>/robot/contacts</topic>
    <link_name>left_foot</link_name>
    <threshold>5.0</threshold>
</plugin>

<plugin name="velocity_controller" filename="libvelocity_control.so">
    <joint_name>left_hip_yaw</joint_name>
    <Kp>100</Kp>
    <Kd>10</Kd>
</plugin>

<plugin name="random_obstacle_spawner" filename="libobstacle_spawner.so">
    <spawn_interval>5.0</spawn_interval>
    <max_obstacles>10</max_obstacles>
    <spawn_area>5 5</spawn_area>
</plugin>
```

### Actors for Animated Elements

Actors simulate moving elements without full physics:

```xml
<actor name="walking_human">
    <skin>
        <filename>file://models/human_animation.dae</filename>
        <scale>1.0</scale>
    </skin>
    <animation name="walking">
        <filename>file://animations/walk.dae</filename>
        <scale>1.0</scale>
        <speed>1.0</speed>
    </animation>
    <script>
        <trajectory id="0" type="walking">
            <waypoint>
                <time>0</time>
                <pose>0 0 0 0 0 0</pose>
            </waypoint>
            <waypoint>
                <time>2</time>
                <pose>1.5 0 0 0 0 0</pose>
            </waypoint>
            <waypoint>
                <time>4</time>
                <pose>3.0 0 0 0 0 0</pose>
            </waypoint>
        </trajectory>
    </script>
</actor>
```

### Population for Mass Object Spawning

Populations efficiently create many similar objects:

```xml
<population name="boxes">
    <model name="test_box">
        <link name="link">
            <visual name="visual">
                <geometry>
                    <box size="0.2 0.2 0.2"/>
                </geometry>
                <material>
                    <script>
                        <uri>file://materials/scripts/gazebo.material</uri>
                        <name>Gazebo/Blue</name>
                    </script>
                </material>
            </visual>
        </link>
    </model>
    <pose>0 2 0.1 0 0 0</pose>
    <box>
        <size>3 2 0.1</size>
        <cell_count>20</cell_count>
    </box>
    <distribution>
        <type>random</type>
    </distribution>
</population>
```

## Key Takeaways

SDF provides the foundation for creating realistic simulation environments:

- **World structure** organizes physics, scene, and models hierarchically
- **Physics configuration** enables tuning for stability and realism
- **Model inclusion** composes complex environments from simpler components
- **Lighting systems** enable realistic camera and navigation simulation
- **Plugins and actors** extend SDF with custom behavior and animation

With simulation environments defined, we can now explore how Unity complements traditional simulation for visualization and development.
