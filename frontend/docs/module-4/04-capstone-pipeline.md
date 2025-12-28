---
sidebar_position: 4
---

# Capstone Pipeline

This capstone module integrates all concepts from the textbook into a comprehensive Physical AI system, demonstrating the complete pipeline from perception to action in a humanoid robotics application. The integrated system brings together voice interfaces, cognitive planning, visual perception, navigation, locomotion, and manipulation into a cohesive robot that can understand natural language commands, plan complex tasks, and execute them in human environments.

## System Architecture Overview

The capstone system architecture follows a modular, hierarchical design that enables flexibility and robustness. Each component can be developed, tested, and improved independently while maintaining clean interfaces with other components. This architecture reflects best practices in robotic system design while demonstrating how the various textbook topics integrate into a functional whole.

### High-Level System Design

The system consists of five primary layers that process information from input to action. The Perception Layer receives and processes sensory data from cameras, microphones, and other sensors to build an understanding of the environment. The Cognition Layer uses this perceptual understanding along with natural language commands to plan high-level actions through LLM-based reasoning. The Planning Layer converts abstract plans into concrete motion plans for navigation and manipulation. The Control Layer executes these plans through low-level controllers for locomotion and manipulation. The Execution Layer monitors execution, handles errors, and provides feedback to higher layers.

```python
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CapstoneSystem")

class SystemState(Enum):
    """Overall system states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    EXECUTING = "executing"
    RECOVERING = "recovering"
    ERROR = "error"

@dataclass
class PerceptionResult:
    """Result from perception processing."""
    timestamp: datetime
    scene_description: str
    detected_objects: List[Dict]
    human_positions: List[Dict]
    obstacles: List[Dict]
    free_space: Dict
    confidence: float

@dataclass
class PlanStep:
    """A single step in a robot plan."""
    step_id: str
    action_type: str
    description: str
    parameters: Dict
    dependencies: List[str]
    estimated_duration: float
    status: str = "pending"

@dataclass
class TaskCommand:
    """High-level task command from user."""
    command_id: str
    raw_text: str
    intent: str
    parameters: Dict
    timestamp: datetime
    priority: int = 0

@dataclass
class SystemStatus:
    """Current system status."""
    state: SystemState
    battery_level: float
    current_pose: Dict
    last_command: Optional[TaskCommand]
    active_plan: Optional[List[PlanStep]]
    recent_perception: Optional[PerceptionResult]
    errors: List[str]

class PhysicalAISystem:
    """
    Complete Physical AI system integrating all textbook concepts.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.state = SystemState.INITIALIZING

        # Component status
        self.status = SystemStatus(
            state=SystemState.INITIALIZING,
            battery_level=100.0,
            current_pose={"x": 0, "y": 0, "theta": 0},
            last_command=None,
            active_plan=None,
            recent_perception=None,
            errors=[]
        )

        # Initialize all subsystems
        self._init_perception()
        self._init_voice()
        self._init_planning()
        self._init_navigation()
        self._init_manipulation()
        self._init_locomotion()
        self._init_safety()

        # Set state to idle
        self.state = SystemState.IDLE
        logger.info("Physical AI System initialized successfully")

    def _init_perception(self):
        """Initialize perception subsystem."""
        # Visual perception (from Module 3 - Visual SLAM)
        self.visual_slam = VisualSLAMInterface(self.config['slam'])
        self.object_detector = ObjectDetector(self.config['detection'])
        self.human_detector = HumanPoseEstimator(self.config['human_pose'])

        # Point cloud processing
        self.point_cloud_processor = PointCloudProcessor(self.config['point_cloud'])

    def _init_voice(self):
        """Initialize voice interface (from Module 4 - Voice to Action)."""
        self.voice_recognizer = WhisperRecognizer(self.config['whisper'])
        self.command_parser = VoiceCommandParser()
        self.tts_player = TextToSpeechPlayer(self.config['tts'])

    def _init_planning(self):
        """Initialize cognitive planning (from Module 4 - LLM Planning)."""
        self.llm_planner = LLMCognitivePlanner(
            llm_client=self._create_llm_client(),
            available_actions=self._get_available_actions()
        )
        self.safety_validator = SafetyValidator(self.config['safety'])
        self.plan_executor = PlanExecutor()

    def _init_navigation(self):
        """Initialize navigation (from Module 3 - Nav2)."""
        self.nav2_interface = Nav2Interface(self.config['nav2'])
        self.global_planner = AStarPlanner(self.config['global_planner'])
        self.local_planner = DWAPlanner(self.config['local_planner'])
        self.costmap_manager = CostmapManager(self.config['costmap'])

    def _init_manipulation(self):
        """Initialize manipulation system."""
        self.manipulation_planner = ManipulationPlanner(self.config['manipulation'])
        self.grasp_planner = GraspPlanner(self.config['grasp'])
        self.ik_solver = InverseKinematicsSolver(self.config['ik'])

    def _init_locomotion(self):
        """Initialize locomotion (from Module 4 - Humanoid Locomotion)."""
        self.walking_controller = WalkingController(self.config['walking'])
        self.balance_controller = BalanceMPC(self.config['balance'])
        self.trajectory_generator = WalkingPatternGenerator(self.config['trajectory'])

    def _init_safety(self):
        """Initialize safety monitoring."""
        self.safety_monitor = SafetyMonitor(self.config['safety'])
        self.emergency_stop = EmergencyStop()

    def _create_llm_client(self):
        """Create LLM client for cognitive planning."""
        from openai import OpenAI
        return OpenAI(api_key=self.config['openai_api_key'])

    def _get_available_actions(self) -> List[Dict]:
        """Get list of available robot actions for planning."""
        return [
            {
                "name": "navigate_to",
                "description": "Navigate to a target location",
                "parameters": {
                    "destination": {"type": "string", "description": "Target location name"},
                    "position": {"type": "object", "description": "Optional XYZ coordinates"}
                }
            },
            {
                "name": "pick_up",
                "description": "Pick up an object",
                "parameters": {
                    "object_name": {"type": "string", "description": "Object to pick up"},
                    "object_id": {"type": "string", "description": "Optional specific object ID"}
                }
            },
            {
                "name": "place_down",
                "description": "Place held object at a location",
                "parameters": {
                    "location": {"type": "string", "description": "Where to place the object"}
                }
            },
            {
                "name": "look_at",
                "description": "Direct gaze toward a target",
                "parameters": {
                    "target": {"type": "string", "description": "Target to look at"}
                }
            },
            {
                "name": "scan_environment",
                "description": "Scan environment to detect objects",
                "parameters": {}
            },
            {
                "name": "locate_object",
                "description": "Search for a specific object",
                "parameters": {
                    "object_name": {"type": "string", "description": "Object to find"}
                }
            },
            {
                "name": "get_state",
                "description": "Get current robot state",
                "parameters": {}
            },
            {
                "name": "check_battery",
                "description": "Check battery level",
                "parameters": {}
            }
        ]

    async def process_voice_command(self, audio_data: np.ndarray) -> str:
        """
        Process a voice command and execute corresponding task.

        This is the main entry point for voice-controlled operation.
        """
        self.state = SystemState.PERCEIVING
        self.status.state = SystemState.PERCEIVING

        try:
            # Step 1: Transcribe voice to text
            transcription = await self._transcribe_voice(audio_data)
            logger.info(f"Transcribed: {transcription}")

            # Step 2: Parse command intent
            command = self._parse_command(transcription)
            self.status.last_command = command

            # Step 3: Get current environment state
            perception = await self._perceive_environment()
            self.status.recent_perception = perception

            # Step 4: Generate plan using LLM
            self.state = SystemState.PLANNING
            self.status.state = SystemState.PLANNING
            plan = self._generate_plan(command, perception)
            self.status.active_plan = plan.steps

            # Step 5: Validate plan for safety
            validation = self.safety_validator.validate_plan(plan)
            if not validation['valid']:
                await self._handle_invalid_plan(validation)
                return "Cannot execute: " + "; ".join(validation['errors'])

            # Step 6: Execute plan
            self.state = SystemState.EXECUTING
            self.status.state = SystemState.EXECUTING
            result = await self._execute_plan(plan)

            logger.info(f"Task completed: {result}")
            await self._speak("Task completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self.status.errors.append(str(e))
            self.state = SystemState.ERROR
            await self._speak("I encountered an error. Please try again.")
            return f"Error: {str(e)}"

        finally:
            self.state = SystemState.IDLE
            self.status.state = SystemState.IDLE

    async def _transcribe_voice(self, audio_data: np.ndarray) -> str:
        """Transcribe voice audio to text."""
        result = self.voice_recognizer.transcribe(audio_data)
        return result["text"]

    def _parse_command(self, transcription: str) -> TaskCommand:
        """Parse transcribed text into structured command."""
        parsed = self.command_parser.parse(transcription)

        return TaskCommand(
            command_id=f"cmd_{datetime.now().timestamp()}",
            raw_text=transcription,
            intent=parsed.intent.value,
            parameters=self.command_parser.entities_to_action(parsed),
            timestamp=datetime.now()
        )

    async def _perceive_environment(self) -> PerceptionResult:
        """Perceive the current environment."""
        # Get visual data
        rgb_image = self.visual_slam.get_current_image()
        point_cloud = self.visual_slam.get_point_cloud()

        # Detect objects
        detections = self.object_detector.detect(rgb_image)

        # Detect humans
        humans = self.human_detector.detect_poses(rgb_image)

        # Build scene description
        scene_desc = self._build_scene_description(detections, humans)

        # Process point cloud for obstacle detection
        obstacles = self.point_cloud_processor.detect_obstacles(point_cloud)
        free_space = self.point_cloud_processor.compute_free_space(point_cloud)

        return PerceptionResult(
            timestamp=datetime.now(),
            scene_description=scene_desc,
            detected_objects=detections,
            human_positions=humans,
            obstacles=obstacles,
            free_space=free_space,
            confidence=0.85
        )

    def _build_scene_description(self, detections: List, humans: List) -> str:
        """Build natural language scene description."""
        desc_parts = []

        if len(detections) > 0:
            obj_names = [d['class_name'] for d in detections]
            desc_parts.append(f"I see {', '.join(obj_names)}")

        if len(humans) > 0:
            desc_parts.append(f"and {len(humans)} {'person' if len(humans) == 1 else 'people'}")

        return " ".join(desc_parts) if desc_parts else "The area appears clear"

    def _generate_plan(self, command: TaskCommand,
                       perception: PerceptionResult) -> Plan:
        """Generate execution plan using LLM."""
        # Build context for LLM
        context = f"""
Current perception: {perception.scene_description}
Detected objects: {[d['class_name'] for d in perception.detected_objects]}
Humans detected: {len(perception.human_positions)}
Robot position: {self.status.current_pose}
Battery: {self.status.battery_level}%

Command: {command.raw_text}
Intent: {command.intent}
Parameters: {command.parameters}
"""

        # Generate plan
        plan = self.llm_planner.generate_plan(
            goal=command.raw_text,
            context=context
        )

        # Enrich plan with additional details
        for step in plan.steps:
            step.estimated_duration = self._estimate_step_duration(step)

        return plan

    def _estimate_step_duration(self, step: PlanStep) -> float:
        """Estimate duration for a plan step."""
        duration_estimates = {
            "navigate_to": 5.0,  # seconds per meter
            "pick_up": 3.0,
            "place_down": 2.0,
            "look_at": 1.0,
            "scan_environment": 2.0,
            "locate_object": 3.0
        }
        return duration_estimates.get(step.action_type, 2.0)

    async def _execute_plan(self, plan: Plan) -> str:
        """Execute the generated plan."""
        completed_steps = []
        failed_steps = []

        for step in plan.steps:
            # Check dependencies
            for dep in step.dependencies:
                if dep not in [s.step_id for s in completed_steps]:
                    failed_steps.append((step, f"Missing dependency: {dep}"))
                    continue

            # Execute step
            try:
                result = await self._execute_step(step)
                step.status = "completed"
                completed_steps.append(step.step_id)
                logger.info(f"Completed step: {step.description}")

            except Exception as e:
                step.status = "failed"
                failed_steps.append((step, str(e)))
                logger.error(f"Failed step {step.step_id}: {e}")

                # Try recovery
                recovered = await self._attempt_recovery(step, e)
                if not recovered:
                    return f"Task failed at step: {step.description}"

        if failed_steps:
            return f"Completed with {len(failed_steps)} failures"
        return "All steps completed successfully"

    async def _execute_step(self, step: PlanStep) -> Dict:
        """Execute a single plan step."""
        if step.action_type == "navigate_to":
            return await self._execute_navigation(step)

        elif step.action_type == "pick_up":
            return await self._execute_pickup(step)

        elif step.action_type == "place_down":
            return await self._execute_placedown(step)

        elif step.action_type == "look_at":
            return await self._execute_lookat(step)

        elif step.action_type == "scan_environment":
            return await self._execute_scan(step)

        elif step.action_type == "locate_object":
            return await self._execute_locate(step)

        else:
            raise ValueError(f"Unknown action type: {step.action_type}")

    async def _execute_navigation(self, step: PlanStep) -> Dict:
        """Execute navigation action."""
        destination = step.parameters.get("destination", "unknown")
        position = step.parameters.get("position")

        # Plan path
        path = self.nav2_interface.plan_path(
            start=self.status.current_pose,
            goal=position or destination
        )

        # Execute walking
        await self.walking_controller.walk_along_path(path)

        return {"status": "arrived", "destination": destination}

    async def _execute_pickup(self, step: PlanStep) -> Dict:
        """Execute pick up action."""
        object_name = step.parameters.get("object_name", "unknown")
        object_id = step.parameters.get("object_id")

        # Navigate to object
        if object_id:
            object_pose = self.object_detector.get_pose(object_id)
        else:
            object_pose = self.object_detector.find_object(object_name)

        # Navigate close to object
        await self._navigate_to(object_pose['position'], approach=True)

        # Plan grasp
        grasp = self.grasp_planer.plan_grasp(object_pose)

        # Execute grasp
        success = await self.manipulation_planer.execute_grasp(grasp)

        if success:
            return {"status": "grasped", "object": object_name}
        else:
            raise RuntimeError(f"Failed to grasp {object_name}")

    async def _execute_placedown(self, step: PlanStep) -> Dict:
        """Execute place down action."""
        location = step.parameters.get("location", "designated spot")

        # Get placement position
        if location in self.known_locations:
            position = self.known_locations[location]
        else:
            position = self._find_placement_location()

        # Navigate to location
        await self._navigate_to(position)

        # Place object
        await self.manipulation_planer.place_object(position)

        return {"status": "placed", "location": location}

    async def _execute_lookat(self, step: PlanStep) -> Dict:
        """Execute look at action."""
        target = step.parameters.get("target", "forward")

        # Determine target position
        if target in self.known_locations:
            target_pos = self.known_locations[target]
        elif target.lower() == "person":
            target_pos = self._find_nearest_person()
        else:
            target_pos = self._parse_target_direction(target)

        # Move head
        await self.manipulation_planer.look_at(target_pos)

        return {"status": "looking", "target": target}

    async def _execute_scan(self, step: PlanStep) -> Dict:
        """Execute environment scan."""
        # Rotate and collect observations
        observations = []

        for angle in [0, 90, 180, 270]:
            await self.manipulation_planer.rotate_head(angle)
            obs = await self._perceive_environment()
            observations.append(obs)

        # Merge observations
        merged = self._merge_observations(observations)

        return {"status": "scanned", "observations": merged}

    async def _execute_locate(self, step: PlanStep) -> Dict:
        """Execute locate object action."""
        object_name = step.parameters.get("object_name", "unknown")

        # Scan environment
        await self._execute_scan(step)

        # Search for object
        results = self.object_detector.search(object_name)

        if results:
            return {"status": "found", "object": object_name, "location": results[0]}
        else:
            return {"status": "not_found", "object": object_name}

    async def _attempt_recovery(self, step: PlanStep, error: Exception) -> bool:
        """Attempt recovery from step failure."""
        self.state = SystemState.RECOVERING
        self.status.state = SystemState.RECOVERING

        logger.info(f"Attempting recovery from error: {error}")

        # Try simple recovery strategies
        if "navigation" in step.action_type:
            # Retry with different approach
            await asyncio.sleep(1)
            return True

        elif "grasp" in step.action_type:
            # Try different grasp strategy
            return True

        return False

    async def _navigate_to(self, position: Dict, approach: bool = False):
        """Navigate to a position."""
        target_pose = {
            "x": position.get("x", 0),
            "y": position.get("y", 0),
            "theta": position.get("theta", 0)
        }

        if approach:
            # Approach from a distance
            approach_pose = {
                "x": target_pose["x"] - 0.5,
                "y": target_pose["y"],
                "theta": target_pose["theta"]
            }
            await self.walking_controller.walk_to(approach_pose)

        await self.walking_controller.walk_to(target_pose)

    async def _speak(self, text: str):
        """Speak text through TTS."""
        await self.tts_player.speak(text)

    async def _handle_invalid_plan(self, validation: Dict):
        """Handle invalid plan with suggestions."""
        await self._speak(
            f"I cannot complete this task. Issues: {', '.join(validation['errors'])}"
        )

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        return self.status

    def emergency_shutdown(self):
        """Emergency shutdown all systems."""
        logger.warning("Emergency shutdown initiated")
        self.emergency_stop.trigger()
        self.walking_controller.stop()
        self.manipulation_planer.release()
```

## Perception Pipeline Integration

The perception pipeline integrates multiple sensor modalities to build a comprehensive understanding of the environment. Visual data from cameras provides rich semantic information about objects and humans, while point cloud data from depth sensors enables precise 3D localization.

```python
class PerceptionPipeline:
    """
    Integrated perception pipeline for the capstone system.
    """

    def __init__(self, config: Dict):
        self.camera = RGBDCamera(config['camera'])
        self.lidar = LidarSensor(config['lidar'])
        self.imu = IMUSensor(config['imu'])

        # Processing modules
        self.slam = VisualSLAM()
        self.segmentation = SemanticSegmentor()
        self.detection = ObjectDetector()
        self.tracking = MultiObjectTracker()

    async def process(self) -> PerceptionResult:
        """Process current sensor data into perception result."""
        # Capture synchronized sensor data
        rgb, depth = await self.camera.capture()
        point_cloud = await self.lidar.capture()
        imu_data = await self.imu.get_data()

        # Run visual SLAM
        pose, map_points = self.slam.process_frame(rgb, depth)

        # Semantic segmentation
        segments = self.segmentation.segment(rgb)

        # Object detection
        detections = self.detection.detect(rgb)

        # Multi-object tracking
        tracked_objects = self.tracking.update(detections, point_cloud)

        # Build perception result
        result = PerceptionResult(
            timestamp=datetime.now(),
            scene_description=self._describe_scene(tracked_objects),
            detected_objects=tracked_objects,
            human_positions=self._extract_humans(tracked_objects),
            obstacles=self._detect_obstacles(point_cloud, segments),
            free_space=self._compute_free_space(point_cloud, segments),
            confidence=self._compute_confidence(tracked_objects)
        )

        return result

    def _describe_scene(self, objects: List) -> str:
        """Generate natural language scene description."""
        if not objects:
            return "The area appears clear with no significant objects detected."

        categories = {}
        for obj in objects:
            cat = obj.category
            categories[cat] = categories.get(cat, 0) + 1

        parts = []
        for cat, count in categories.items():
            if count == 1:
                parts.append(f"a {cat}")
            else:
                parts.append(f"{count} {cat}s")

        return "I see " + ", ".join(parts) + "."

    def _extract_humans(self, objects: List) -> List[Dict]:
        """Extract human positions from detected objects."""
        humans = []
        for obj in objects:
            if obj.category == "person":
                humans.append({
                    "position": obj.position_3d,
                    "orientation": obj.orientation,
                    "pose": obj.pose if hasattr(obj, 'pose') else None
                })
        return humans

    def _detect_obstacles(self, point_cloud: np.ndarray,
                          segments: np.ndarray) -> List[Dict]:
        """Detect obstacles from point cloud."""
        obstacles = []

        # Ground plane removal
        ground_mask = self._remove_ground_plane(point_cloud)

        # Find obstacle points
        obstacle_points = point_cloud[~ground_mask]

        # Cluster obstacle points
        clusters = self._cluster_points(obstacle_points)

        for cluster in clusters:
            if cluster.size > 100:  # Minimum points for obstacle
                center = np.mean(cluster, axis=0)
                extent = np.ptp(cluster, axis=0)

                obstacles.append({
                    "position": {"x": center[0], "y": center[1], "z": center[2]},
                    "size": {"x": extent[0], "y": extent[1], "z": extent[2]},
                    "type": "obstacle"
                })

        return obstacles

    def _compute_free_space(self, point_cloud: np.ndarray,
                            segments: np.ndarray) -> Dict:
        """Compute traversable free space."""
        # Project to 2D occupancy grid
        grid = np.zeros((100, 100))  # 10m x 10m grid at 0.1m resolution

        # Mark occupied cells
        for point in point_cloud:
            if point[2] < 2.0:  # Below head height
                grid_x = int((point[0] + 5) / 0.1)
                grid_y = int((point[1] + 5) / 0.1)
                if 0 <= grid_x < 100 and 0 <= grid_y < 100:
                    grid[grid_x, grid_y] = 1

        return {
            "grid": grid,
            "resolution": 0.1,
            "origin": {"x": -5, "y": -5}
        }
```

## Complete Human-Robot Interaction Flow

The complete HRI flow demonstrates how all components work together to enable natural human-robot interaction:

```python
async def demonstrate_interaction():
    """
    Demonstrate complete human-robot interaction flow.
    """
    # Initialize system
    config = load_config("capstone_config.yaml")
    system = PhysicalAISystem(config)

    # Simulate user interaction
    print("=== Human-Robot Interaction Demo ===\n")

    # User gives voice command
    print("User: 'Hey robot, please bring me the water bottle from the table.'")
    audio_command = simulate_voice_input("bring me the water bottle from the table")

    # Process command
    result = await system.process_voice_command(audio_command)

    print(f"\nRobot: Processing command...")
    print(f"  Transcribed: '{result}'")

    # Show system status
    status = system.get_status()
    print(f"\nSystem Status:")
    print(f"  State: {status.state.value}")
    print(f"  Battery: {status.battery_level}%")
    print(f"  Position: {status.current_pose}")

    # Show perception result
    if status.recent_perception:
        print(f"\nPerception:")
        print(f"  {status.recent_perception.scene_description}")
        print(f"  Confidence: {status.recent_perception.confidence:.2f}")

    # Show plan
    if status.active_plan:
        print(f"\nPlan ({len(status.active_plan)} steps):")
        for i, step in enumerate(status.active_plan, 1):
            print(f"  {i}. {step.description} ({step.action_type})")
```

## Testing and Validation

Comprehensive testing ensures system reliability:

```python
class SystemValidator:
    """
    Validates capstone system performance.
    """

    def __init__(self, system: PhysicalAISystem):
        self.system = system

    def run_tests(self) -> Dict:
        """
        Run all validation tests.
        """
        results = {
            "perception_accuracy": self._test_perception(),
            "voice_recognition_accuracy": self._test_voice(),
            "planning_success_rate": self._test_planning(),
            "execution_success_rate": self._test_execution(),
            "safety_compliance": self._test_safety()
        }

        # Compute overall score
        results["overall_score"] = np.mean(list(results.values()))

        return results

    def _test_perception(self) -> float:
        """Test perception accuracy."""
        test_cases = load_test_cases("perception_test_set")

        correct = 0
        for test in test_cases:
            result = self.system._perceive_environment()
            if self._evaluate_perception(result, test.ground_truth):
                correct += 1

        return correct / len(test_cases)

    def _test_voice(self) -> float:
        """Test voice recognition accuracy."""
        test_audio = load_test_cases("voice_test_set")

        correct = 0
        for audio in test_audio:
            result = self.system._transcribe_voice(audio.data)
            if self._evaluate_transcription(result, audio.ground_truth):
                correct += 1

        return correct / len(test_audio)

    def _test_planning(self) -> float:
        """Test planning success rate."""
        test_commands = load_test_cases("planning_test_set")

        success = 0
        for cmd in test_commands:
            plan = self.system._generate_plan(cmd.command, cmd.perception)
            if plan and len(plan.steps) > 0:
                success += 1

        return success / len(test_commands)

    def _test_execution(self) -> float:
        """Test task execution success rate."""
        test_tasks = load_test_cases("execution_test_set")

        completed = 0
        for task in test_tasks:
            result = asyncio.run(
                self.system._execute_plan(task.plan)
            )
            if "failed" not in result.lower():
                completed += 1

        return completed / len(test_tasks)

    def _test_safety(self) -> float:
        """Test safety compliance."""
        dangerous_commands = load_test_cases("safety_test_set")

        blocked = 0
        for cmd in dangerous_commands:
            plan = self.system._generate_plan(cmd.command, cmd.perception)
            validation = self.system.safety_validator.validate_plan(plan)
            if not validation['valid']:
                blocked += 1

        return blocked / len(dangerous_commands)
```

## Key Takeaways

The capstone pipeline demonstrates how all textbook concepts integrate into a comprehensive Physical AI system:

- **Modular architecture** enables independent development and testing of components
- **Voice interface** provides natural interaction for commanding the robot
- **LLM planning** enables high-level reasoning and task decomposition
- **Perception pipeline** integrates multiple sensors for environment understanding
- **Navigation and locomotion** enable the robot to move through environments
- **Manipulation** enables object interaction and task completion
- **Safety systems** ensure reliable and safe operation
- **Testing framework** validates system performance across scenarios

This completes the Physical AI Foundations textbook. The knowledge gained here provides a strong foundation for developing advanced humanoid robots capable of operating in human environments.
