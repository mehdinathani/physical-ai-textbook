---
sidebar_position: 2
---

# Python rclpy Agents

ROS 2's Python client library, rclpy, enables rapid development of robotic applications using Python's expressive syntax and extensive ecosystem. This chapter explores the creation of intelligent agents using rclpy, covering fundamental concepts, design patterns, and implementation strategies that form the foundation of ROS 2-based robot applications.

## Understanding rclpy Architecture

rclpy is the Python binding for ROS 2's client library, rcl (ROS Client Library). Unlike ROS 1, where rospy was a separate implementation, rclpy shares core functionality with rclcpp (the C++ client library). This ensures consistent behavior across language bindings and enables features like distributed development and real-time guarantees.

### The ROS 2 Software Stack

To understand rclpy, you must first understand the layers it builds upon:

**RCL (ROS Client Library)**: The language-agnostic core that implements ROS concepts—nodes, topics, services, actions, parameters, and lifecycle management. RCL handles the complexity of DDS communication, discovery, and message serialization.

**RCLPY (Python Client Library)**: A Python wrapper around RCL that provides Pythonic interfaces for all ROS concepts. RCLPY enables Python developers to work with familiar constructs while leveraging RCL's robust communication layer.

**RMW (ROS Middleware Interface)**: The abstract interface to DDS implementations. Multiple RMW implementations exist, from simple intra-process communication to full DDS with real-time extensions. RCLPY works with any conforming RMW implementation.

**DDS (Data Distribution Service)**: The underlying publish-subscribe middleware. DDS handles discovery, message transport, quality of service (QoS), and reliability. ROS 2 uses DDS as its default middleware, providing enterprise-grade communication capabilities.

### Nodes: The Fundamental Building Block

A ROS 2 node is a process that performs computation. Nodes are the fundamental unit of organization in a ROS 2 system—they encapsulate functionality, communicate with other nodes, and can be composed into complex applications.

```python
import rclpy
from rclpy.node import Node

class BasicNode(Node):
    """A minimal ROS 2 node demonstrating basic structure."""

    def __init__(self):
        super().__init__('basic_node')
        # Node initialization happens here
        self.get_logger().info('Basic node has been initialized')

        # Example: Create a timer for periodic operations
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        """Called every 1.0 seconds."""
        self.counter += 1
        self.get_logger().info(f'Counter: {self.counter}')

def main(args=None):
    """Entry point for the node."""
    rclpy.init(args=args)
    node = BasicNode()

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

This example demonstrates several key concepts:

1. **Node Inheritance**: Custom nodes inherit from `rclpy.node.Node`, gaining all node functionality.
2. **Node Naming**: Each node has a unique name within its ROS domain.
3. **Logging**: The `get_logger()` method provides ROS-compliant logging with severity levels.
4. **Timer Creation**: Timers enable periodic callbacks essential for control loops.
5. **Spin Loop**: `rclpy.spin()` keeps the node alive and processes incoming messages.

### The Executor Model

ROS 2 uses an executor model to process incoming events—messages, timers, service requests—in a structured way. The executor calls back to registered handlers when events occur:

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.subscription = self.create_subscription(
            sensor_msgs.msg.Image,
            'camera/image',
            self.image_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

    def image_callback(self, msg):
        # Process incoming image
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')

def main():
    rclpy.init()
    node = SensorNode()

    # Single-threaded executor processes one event at a time
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
```

Multi-threaded and static single-threaded executors provide options for different concurrency requirements. The choice affects how your node handles incoming messages when processing is slow.

## Publishers and Subscribers

The publish-subscribe pattern is the most common communication pattern in ROS 2. Publishers send messages to topics; subscribers receive messages from topics they're interested in.

### Creating Publishers

Publishers broadcast messages to all interested subscribers:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class VelocityPublisher(Node):
    """Publishes velocity commands for a robot."""

    def __init__(self):
        super().__init__('velocity_publisher')

        # Create a publisher with message type, topic name, and QoS profile
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10  # QoS depth
        )

        # Create a string publisher for status messages
        self.status_pub = self.create_publisher(
            String,
            'robot_status',
            10
        )

        # Timer for periodic command generation
        self.timer = self.create_timer(0.1, self.publish_velocity)
        self.angle = 0.0

    def publish_velocity(self):
        """Generate and publish velocity commands."""
        msg = Twist()

        # Create a circular motion pattern
        msg.linear.x = 0.5  # Forward speed
        msg.angular.z = 0.5  # Turning rate

        self.cmd_vel_pub.publish(msg)

        # Also publish status
        status = String()
        status.data = f'Publishing: x={msg.linear.x:.2f}, z={msg.angular.z:.2f}'
        self.status_pub.publish(status)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Key publisher concepts include:

- **Message Types**: Publishers specify their message type at creation. ROS 2 checks type compatibility at compile time (for statically typed languages) and runtime (for Python).
- **QoS Profiles**: The depth (10 in the example) controls how many messages to buffer. Different profiles offer reliability, durability, and deadline guarantees.
- **Publishing**: Call `publish(message)` to send a message. The message object must match the declared type.

### Creating Subscribers

Subscribers receive messages from topics:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class NavigationNode(Node):
    """Processes sensor data for navigation."""

    def __init__(self):
        super().__init__('navigation_node')

        # Subscribe to laser scan for obstacle detection
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        # Subscribe to goal poses
        self.goal_sub = self.create_subscription(
            PoseStamped,
            'goal',
            self.goal_callback,
            10
        )

        # Subscribe to odometry for localization
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # Store latest data
        self.latest_scan = None
        self.latest_goal = None
        self.latest_odom = None

    def scan_callback(self, msg: LaserScan):
        """Process laser scan data."""
        self.latest_scan = msg

        # Find closest obstacle
        if msg.ranges:
            min_range = min(msg.ranges)
            min_index = msg.ranges.index(min_range)
            min_angle = msg.angle_min + min_index * msg.angle_increment

            self.get_logger().info(
                f'Closest obstacle: {min_range:.2f}m at {min_angle:.2f}rad'
            )

    def goal_callback(self, msg: PoseStamped):
        """Process new navigation goal."""
        self.latest_goal = msg
        self.get_logger().info(
            f'New goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )

    def odom_callback(self, msg: Odometry):
        """Process odometry update."""
        self.latest_odom = msg
        # Could update pose estimate here

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Subscriber callbacks receive messages directly when they arrive. Important considerations include:

- **Callback Execution**: Subscriber callbacks execute in the executor's thread. Long-running callbacks block other processing.
- **Message Retention**: ROS 2 doesn't retain messages by default. Subscribers only receive messages published after subscription.
- **Type Hints**: Using type hints (like `msg: LaserScan`) improves code clarity and enables static analysis.

### Custom Message Types

For domain-specific communication, create custom messages:

```python
# human_robot_interaction/msg/HumanAction.msg
string action_type
string description
float64 confidence
string[] related_objects
geometry_msgs/PoseStamped human_pose
geometry_msgs/PoseStamped robot_pose
time timestamp
```

Custom messages compile to Python classes with convenient accessors. In rclpy:

```python
from human_robot_interaction.msg import HumanAction

# Publish custom message
action = HumanAction()
action.action_type = 'reaching'
action.confidence = 0.85
action.description = 'Human is reaching for cup'
action.human_pose = detected_pose
publisher.publish(action)
```

## Services and Actions

While publish-subscribe handles continuous data streams, services and actions handle request-response interactions.

### Services

Services provide synchronous request-response communication:

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from my_robot_package.srv import MoveToPose

class MotionService(Node):
    """Provides motion services for the robot."""

    def __init__(self):
        super().__init__('motion_service')

        # Create service server
        self.move_service = self.create_service(
            MoveToPose,
            'move_to_pose',
            self.move_callback
        )

        self.enable_service = self.create_service(
            SetBool,
            'enable_motors',
            self.enable_callback
        )

        self.emergency_stop_service = self.create_service(
            Trigger,
            'emergency_stop',
            self.e_stop_callback
        )

        self.motors_enabled = False

    def move_callback(self, request, response):
        """Handle move_to_pose service request."""
        if not self.motors_enabled:
            response.success = False
            response.message = 'Motors are disabled'
            return response

        self.get_logger().info(
            f'Moving to ({request.x}, {request.y}, {request.theta})'
        )

        # In a real system, this would trigger motion planning
        # For now, we simulate success
        response.success = True
        response.message = 'Motion completed'
        return response

    def enable_callback(self, request, response):
        """Enable or disable motors."""
        self.motors_enabled = request.data
        response.success = True
        response.message = (
            f'Motors {"enabled" if request.data else "disabled"}'
        )
        return response

    def e_stop_callback(self, request, response):
        """Trigger emergency stop."""
        self.motors_enabled = False
        response.success = True
        response.message = 'Emergency stop triggered'
        return response
```

Service clients call these services:

```python
# Client code to call the service
client = self.create_client(MoveToPose, 'move_to_pose')
while not client.wait_for_service(timeout_sec=1.0):
    self.get_logger().warning('Service not available, waiting...')

request = MoveToPose.Request()
request.x = 1.0
request.y = 0.5
request.theta = 0.0

future = client.call_async(request)
# Must handle the response in a callback or spin until complete
```

### Actions

Actions are asynchronous goal-oriented services with feedback:

```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

class NavigationClient(Node):
    """Action client for navigation."""

    def __init__(self):
        super().__init__('navigation_client')
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

    def send_goal(self, x, y, theta):
        """Send a navigation goal."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Convert theta to quaternion
        import math
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2)

        self.get_logger().info(f'Sending goal to ({x}, {y})')

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal acceptance or rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Distance remaining: {feedback.distance_remaining:.2f}m'
        )

    def get_result_callback(self, future):
        """Handle goal completion."""
        result = future.result().result
        status = future.result().status

        if status == 4:  # STATUS_SUCCEEDED
            self.get_logger().info('Navigation succeeded')
        else:
            self.get_logger().error(f'Navigation failed with status: {status}')
```

Actions are ideal for long-running tasks because they provide:
- **Feedback**: Progress updates during goal execution
- **Result**: Final outcome when the goal completes
- **Cancellation**: Ability to cancel in-progress goals
- **Status**: Current state of the goal (accepted, executing, succeeded, aborted, canceled)

## Agent Design Patterns

Robotic agents combine perception, planning, and action in various architectural patterns.

### Reactive Agents

Reactive agents respond directly to sensor input without maintaining internal state models. This simple but effective approach works for many tasks:

```python
class ReactiveObstacleAvoider(Node):
    """A purely reactive obstacle avoidance agent."""

    def __init__(self):
        super().__init__('reactive_avoider')

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        self.SAFE_DISTANCE = 0.5

    def scan_callback(self, scan):
        """React immediately to obstacles."""
        cmd = Twist()

        # Find closest obstacle in each direction
        left_ranges = scan.ranges[:len(scan.ranges)//3]
        right_ranges = scan.ranges[2*len(scan.ranges)//3:]
        front_ranges = scan.ranges[len(scan.ranges)//3:2*len(scan.ranges)//3]

        front_obstacle = min(front_ranges) < self.SAFE_DISTANCE
        left_obstacle = min(left_ranges) < self.SAFE_DISTANCE
        right_obstacle = min(right_ranges) < self.SAFE_DISTANCE

        if front_obstacle:
            cmd.linear.x = 0.0
            if left_obstacle and not right_obstacle:
                cmd.angular.z = -1.0  # Turn right
            elif right_obstacle and not left_obstacle:
                cmd.angular.z = 1.0   # Turn left
            else:
                cmd.angular.z = 2.0   # Turn around
        else:
            cmd.linear.x = 0.3  # Move forward
            if left_obstacle:
                cmd.angular.z = -0.5
            elif right_obstacle:
                cmd.angular.z = 0.5

        self.cmd_vel_pub.publish(cmd)
```

The strength of reactive systems is simplicity and robustness—they cannot get lost or hold incorrect beliefs. The weakness is their inability to plan or learn.

### Deliberative Agents

Deliberative agents maintain world models and plan their actions:

```python
class DeliberativePlanner(Node):
    """A deliberative planning agent."""

    def __init__(self):
        super().__init__('deliberative_planner')

        # Subscriptions
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, 'goal', self.goal_callback, 10
        )

        # Publishers
        self.plan_pub = self.create_publisher(
            Path, 'plan', 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # State
        self.current_pose = None
        self.goal_pose = None
        self.current_plan = None
        self.plan_execution_index = 0

        # Timer for planning loop
        self.planning_timer = self.create_timer(1.0, self.plan_loop)

    def map_callback(self, msg):
        """Update the world model with new map data."""
        # Process map for planning
        pass

    def odom_callback(self, msg):
        """Update current pose estimate."""
        self.current_pose = msg.pose

    def goal_callback(self, msg):
        """Receive new goal and trigger planning."""
        self.goal_pose = msg.pose
        self.plan_and_execute()

    def plan_and_execute(self):
        """Plan a path to the goal and execute it."""
        if not self.current_pose or not self.goal_pose:
            return

        # Create world model
        world_model = self.build_world_model()

        # Plan path
        path = self.plan_path(world_model, self.current_pose, self.goal_pose)

        if path is None:
            self.get_logger().error('No path found to goal')
            return

        # Publish plan for visualization
        self.current_plan = path
        self.plan_execution_index = 0

        plan_msg = Path()
        plan_msg.poses = path
        self.plan_pub.publish(plan_msg)

        self.get_logger().info(f'Planned path with {len(path)} waypoints')

    def plan_loop(self):
        """Execute the current plan."""
        if not self.current_plan:
            return

        # Move to next waypoint
        if self.plan_execution_index < len(self.current_plan):
            waypoint = self.current_plan[self.plan_execution_index]
            self.execute_waypoint(waypoint)

    def execute_waypoint(self, waypoint):
        """Navigate to a single waypoint."""
        # Simple proportional controller to reach waypoint
        pass
```

### Hybrid Architectures

Hybrid architectures combine reactive and deliberative layers:

```python
class HybridRobotController(Node):
    """
    Hierarchical hybrid architecture.

    Layer 1 (Deliberative): Long-term planning and reasoning
    Layer 2 (Executive): Sequencing and coordination
    Layer 3 (Reactive): Safety behaviors and low-level control
    """

    def __init__(self):
        super().__init__('hybrid_controller')

        # Layer 3: Reactive safety behaviors
        self.safety_sub = self.create_subscription(
            LaserScan, 'scan', self.safety_check, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Layer 2: Executive state machine
        self.state = 'IDLE'
        self.states = ['IDLE', 'PLANNING', 'EXECUTING', 'RECOVERING']

        # Layer 1: Deliberative planning
        self.planner = TaskPlanner()

        self.emergency_stop = False

    def safety_check(self, scan):
        """Layer 3: Reactive safety override."""
        min_range = min(scan.ranges) if scan.ranges else float('inf')

        if min_range < 0.2:  # Very close obstacle
            self.emergency_stop = True
            cmd = Twist()
            cmd.linear.x = -0.1  # Back up
            self.cmd_vel_pub.publish(cmd)
        else:
            self.emergency_stop = False

    def update(self):
        """Main update loop for layered architecture."""

        # Layer 3 has already processed safety

        if self.emergency_stop:
            self.state = 'RECOVERING'
            return

        # Layer 2: State transitions
        if self.state == 'IDLE':
            if self.new_task_received():
                self.state = 'PLANNING'
        elif self.state == 'PLANNING':
            plan = self.planner.create_plan(self.current_task)
            if plan:
                self.current_plan = plan
                self.state = 'EXECUTING'
        elif self.state == 'EXECUTING':
            if self.plan_complete():
                self.state = 'IDLE'
            elif self.plan_failed():
                self.state = 'PLANNING'  # Replan

        # Layer 2: Execute current state
        if self.state == 'EXECUTING':
            self.execute_plan_step()
```

## Parameter Management

ROS 2 parameters provide a standardized way to configure nodes:

```python
class ParameterizedNode(Node):
    """Node demonstrating parameter usage."""

    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with defaults
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('min_distance', 0.5)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('robot_name', 'default_robot')

        # Get parameter values
        self.max_velocity = self.get_parameter('max_velocity').value
        self.min_distance = self.get_parameter('min_distance').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.robot_name = self.get_parameter('robot_name').value

        # Log parameter values
        self.get_logger().info(
            f'Robot: {self.robot_name}, Max velocity: {self.max_velocity}'
        )

        # Create timer with configured rate
        self.timer = self.create_timer(1.0/self.publish_rate, self.timer_callback)

    def timer_callback(self):
        # Use parameters in node logic
        pass

    def update_parameters(self, new_params):
        """Update parameters at runtime."""
        for param_name, value in new_params.items():
            self.set_parameters([Parameter(param_name, value)])
```

Parameters can be set via:
- Command line: `ros2 param set /node_name max_velocity 1.5`
- YAML files: `ros2 launch package config.yaml`
- Services: `ros2 service call /node_name set_parameters`

## Best Practices

### Error Handling

Robust error handling is essential for reliable robots:

```python
class RobustNode(Node):
    """Node demonstrating error handling patterns."""

    def __init__(self):
        super().__init__('robust_node')
        self.subscription = self.create_subscription(
            Image, 'input', self.process_image, 10
        )

    def process_image(self, msg):
        try:
            # Process that might fail
            result = self.unsafe_operation(msg)
            self.publish_result(result)
        except ValueError as e:
            self.get_logger().error(f'Processing error: {e}')
            # Attempt recovery or publish error state
        except Exception as e:
            self.get_logger().exception(f'Unexpected error: {e}')
            # This catches everything else
            raise  # Re-raise if it's truly unexpected
```

### Concurrency Considerations

Python's Global Interpreter Lock (GIL) limits true parallelism, but rclpy's executor model enables effective concurrency:

```python
from rclpy.executors import MultiThreadedExecutor

class ConcurrentNode(Node):
    """Node with awareness of concurrency considerations."""

    def __init__(self):
        super().__init__('concurrent_node')

        # Heavy processing should be offloaded
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.subscription = self.create_subscription(
            Image, 'input', self.image_callback, 10
        )

    def image_callback(self, msg):
        # Don't do heavy processing in callback!
        # Offload to thread pool
        self.executor.submit(self.heavy_processing, msg)

    def heavy_processing(self, msg):
        # This runs in a separate thread
        result = self.process_image(msg)
        # Use timers or futures to publish results
```

### Testing

Test ROS 2 nodes with standard Python testing tools:

```python
import unittest
from std_msgs.msg import String
from my_robot_pkg.my_node import MyNode

class TestMyNode(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.node = MyNode()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_publishes_correctly(self):
        # Create a test subscriber
        received_messages = []

        def callback(msg):
            received_messages.append(msg)

        test_sub = self.node.create_subscription(
            String, 'test_topic', callback, 10
        )

        # Publish a test message
        test_pub = self.node.create_publisher(String, 'input', 10)
        test_pub.publish(String(data='test'))

        # Spin briefly to process messages
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0].data, 'test')
```

## Key Takeaways

rclpy provides a powerful Python interface to ROS 2's communication and execution framework. Understanding these fundamentals enables rapid development of sophisticated robotic agents:

- **Nodes** are the fundamental unit of organization—each handles a specific piece of functionality
- **Publishers/Subscribers** handle continuous data streams efficiently
- **Services** provide request-response communication for synchronous operations
- **Actions** support long-running goals with feedback and cancellation
- **Parameters** enable flexible configuration without code changes
- **Design patterns**—reactive, deliberative, hybrid—address different requirements
- **Robustness** requires careful error handling and concurrency management

With these fundamentals established, we can now explore how ROS 2 represents robot geometry through URDF—the topic of the next chapter.
