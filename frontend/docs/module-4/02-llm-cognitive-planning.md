---
sidebar_position: 2
---

# LLM Cognitive Planning

Large Language Models have emerged as powerful cognitive engines for robotic systems, capable of high-level reasoning, task decomposition, and planning that goes far beyond traditional rule-based approaches. By leveraging the vast knowledge encoded in LLMs during pretraining, robots can understand complex natural language instructions, reason about novel situations, and generate appropriate action sequences. This chapter explores the integration of LLMs with robotic planning systems, covering architectural patterns, implementation strategies, and practical considerations for deploying LLM-based cognitive planners.

## Foundations of LLM-Based Robotics

The integration of Large Language Models into robotic systems represents a fundamental shift in how robots approach reasoning and planning. Traditional robotic planning relied on carefully hand-crafted models of the environment and predefined action libraries, limiting robots to domains that could be explicitly modeled. LLMs, trained on massive corpora of text from diverse domains, encode implicit knowledge about the physical world, human behavior, and task structures that can be leveraged for robot planning.

### The LLM as a Cognitive Planner

An LLM-based cognitive planner operates by processing natural language inputs describing tasks, goals, or observations, and generating text outputs that specify robot actions, plans, or reasoning. The fundamental mechanism is autoregressive text generation: given an input prompt describing the current situation and task, the LLM generates tokens one at a time, each conditioned on the previous tokens. Through appropriate prompt engineering, this generation process can be guided to produce structured plans that robots can execute.

The effectiveness of LLMs for robotics stems from several key capabilities. First, LLMs possess extensive world knowledge about objects, their properties, and typical human environments. When asked to plan a task like "clean up the kitchen," LLMs can draw on knowledge about where cleaning supplies are typically located, what sequences of actions make sense, and how to handle various kitchen items. Second, LLMs can perform chain-of-thought reasoning, breaking down complex goals into manageable subtasks through explicit reasoning steps. Third, through fine-tuning or prompt engineering, LLMs can learn to interact with tools and APIs, enabling them to take actions in the physical world.

### Cognitive Architecture Overview

A complete LLM-based cognitive architecture for robotics consists of several interconnected components that enable effective planning and execution. The perception module processes sensor data to build a representation of the current environment state, which is then encoded into a form suitable for the LLM. The working memory maintains the current context, including recent observations, active goals, and partial plans. The LLM itself serves as the reasoning engine, generating plans and decisions based on the current context. Finally, the execution module translates LLM outputs into robot actions and monitors their execution.

```python
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import time

class TaskStatus(Enum):
    """Status of a task in the execution pipeline."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Represents a single task or subtask."""
    id: str
    description: str
    action_type: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "result": self.result,
            "error": self.error
        }

@dataclass
class EnvironmentState:
    """Current state of the robot's environment."""
    robot_position: Optional[Dict[str, float]] = None
    robot_battery: float = 100.0
    held_object: Optional[str] = None
    visible_objects: List[str] = field(default_factory=list)
    known_locations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recent_events: List[Dict] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Convert state to string for LLM prompt."""
        lines = ["Current environment state:"]
        if self.robot_position:
            lines.append(f"Robot position: {self.robot_position}")
        lines.append(f"Battery level: {self.robot_battery}%")
        if self.held_object:
            lines.append(f"Holding: {self.held_object}")
        if self.visible_objects:
            lines.append(f"Visible objects: {', '.join(self.visible_objects)}")
        if self.known_locations:
            lines.append("Known locations:")
            for loc, pos in self.known_locations.items():
                lines.append(f"  - {loc}: {pos}")
        return "\n".join(lines)

@dataclass
class Plan:
    """Represents a complete plan with multiple tasks."""
    id: str
    goal: str
    tasks: List[Task]
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "status": self.status.value,
            "created_at": self.created_at.isoformat()
        }

class LLMCognitivePlanner:
    """
    LLM-based cognitive planner for robot task planning.
    """

    def __init__(self, llm_client, available_actions: List[Dict]):
        self.llm = llm_client
        self.available_actions = available_actions
        self.working_memory = {
            "current_plan": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "environment_state": EnvironmentState()
        }

        # Build action schema for LLM
        self.action_schema = self._build_action_schema()

    def _build_action_schema(self) -> str:
        """Build a description of available actions for the LLM."""
        schema_lines = ["Available robot actions:"]
        for action in self.available_actions:
            schema_lines.append(f"\n- {action['name']}:")
            schema_lines.append(f"  Description: {action['description']}")
            schema_lines.append(f"  Parameters:")
            for param, param_info in action.get("parameters", {}).items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                schema_lines.append(f"    - {param} ({param_type}): {param_desc}")
            schema_lines.append(f"  Returns:")
            for return_field, return_desc in action.get("returns", {}).items():
                schema_lines.append(f"    - {return_field}: {return_desc}")
        return "\n".join(schema_lines)

    def generate_plan(self, goal: str, context: str = "") -> Plan:
        """
        Generate a plan to achieve the given goal.

        Args:
            goal: Natural language description of the goal
            context: Additional context about the environment

        Returns:
            Plan object containing tasks
        """
        prompt = self._build_planning_prompt(goal, context)
        response = self.llm.generate(prompt, max_tokens=2048)

        # Parse the response into a Plan
        plan = self._parse_plan_response(response, goal)
        return plan

    def _build_planning_prompt(self, goal: str, context: str) -> str:
        """Build the planning prompt for the LLM."""
        state_context = self.working_memory["environment_state"].to_context_string()

        prompt = f"""You are a cognitive planning system for a humanoid robot.
Your task is to generate a detailed plan to achieve the given goal.

{state_context}

{context}

GOAL: {goal}

Available actions:
{self.action_schema}

Please generate a step-by-step plan. For each step, specify:
1. Action name
2. Parameters in JSON format
3. Reasoning for this step

Format each step as:
### Step <n>
**Action**: <action_name>
**Parameters**: <JSON object>
**Reasoning**: <brief explanation>

After listing all steps, summarize the plan as a JSON array of tasks in this format:
\`\`\`json
[
  {{
    "id": "step_1",
    "description": "<brief description>",
    "action_type": "<action_name>",
    "parameters": <JSON parameters>,
    "dependencies": []
  }},
  ...
]
\`\`\`
"""
        return prompt

    def _parse_plan_response(self, response: str, goal: str) -> Plan:
        """Parse LLM response into a Plan object."""
        import uuid

        # Extract JSON from response
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if not json_match:
            raise ValueError("Could not extract plan JSON from LLM response")

        tasks_data = json.loads(json_match.group(1))

        tasks = []
        for task_data in tasks_data:
            task = Task(
                id=task_data["id"],
                description=task_data["description"],
                action_type=task_data["action_type"],
                parameters=task_data.get("parameters", {}),
                dependencies=task_data.get("dependencies", [])
            )
            tasks.append(task)

        return Plan(
            id=str(uuid.uuid4()),
            goal=goal,
            tasks=tasks,
            created_at=datetime.now()
        )

    def update_environment_state(self, updates: Dict):
        """Update the working memory with new observations."""
        state = self.working_memory["environment_state"]
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

        # Add to recent events
        self.working_memory["environment_state"].recent_events.append({
            "timestamp": datetime.now().isoformat(),
            "updates": updates
        })

    def explain_plan(self, plan: Plan) -> str:
        """Generate a natural language explanation of a plan."""
        prompt = f"""Explain the following robot plan in natural language.
Focus on what the robot will do and why.

Plan goal: {plan.goal}

Tasks:
"""
        for task in plan.tasks:
            prompt += f"- {task.description} ({task.action_type})\n"

        prompt += "\nProvide a concise explanation of what the robot will do."

        return self.llm.generate(prompt, max_tokens=512)
```

## Tool Use and Function Calling

Modern LLMs can interact with external tools and APIs through structured function calling, enabling them to take actions in the physical world and gather information about the environment. This capability is essential for embodied AI systems that must both reason about and act in the real world.

### Function Calling Architecture

The function calling architecture consists of three main components: the function definitions that specify what tools are available, the function calling mechanism that invokes tools based on LLM output, and the result processing that feeds tool outputs back to the LLM. This creates a loop where the LLM reasons, calls functions to gather information or take actions, receives results, and continues reasoning.

```python
from typing import Callable, Dict, List, Any, Optional
import json
import inspect

class ToolRegistry:
    """
    Registry of available tools for the LLM to call.
    """

    def __init__(self):
        self.tools = {}
        self.tool_functions = {}

    def register(self, name: str, description: str,
                 parameters: Dict, func: Callable):
        """
        Register a tool with the LLM.

        Args:
            name: Unique identifier for the tool
            description: Natural language description
            parameters: Dict of parameter specifications
            func: Python function implementing the tool
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        self.tool_functions[name] = func

    def get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for LLM function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param_name: {
                                "type": param_info.get("type", "string"),
                                "description": param_info.get("description", "")
                            }
                            for param_name, param_info in info["parameters"].items()
                        },
                        "required": list(info["parameters"].keys())
                    }
                }
            }
            for name, info in self.tools.items()
        ]

    def execute(self, name: str, arguments: Dict) -> Any:
        """
        Execute a tool by name with given arguments.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if name not in self.tool_functions:
            raise ValueError(f"Unknown tool: {name}")

        func = self.tool_functions[name]
        return func(**arguments)

class ToolUsingLLM:
    """
    LLM client with tool calling capabilities.
    """

    def __init__(self, base_client, tool_registry: ToolRegistry):
        self.client = base_client
        self.tools = tool_registry

    def generate_with_tools(self, prompt: str,
                            max_iterations: int = 5) -> Dict:
        """
        Generate response with tool calling.

        Args:
            prompt: User prompt
            max_iterations: Maximum tool call iterations

        Returns:
            Dict with 'response' and 'tool_calls' info
        """
        messages = [{"role": "user", "content": prompt}]
        tool_calls = []

        for iteration in range(max_iterations):
            # Get LLM response
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools.get_tool_definitions(),
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            tool_calls.extend(response_message.tool_calls or [])

            if not response_message.tool_calls:
                # No more tool calls, return final response
                return {
                    "response": response_message.content,
                    "tool_calls": tool_calls
                }

            # Execute tool calls
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                try:
                    result = self.tools.execute(function_name, arguments)
                except Exception as e:
                    result = f"Error: {str(e)}"

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        return {
            "response": "Maximum iterations reached",
            "tool_calls": tool_calls
        }

# Example tool definitions for robotics
def create_robot_tools(navigation_system, manipulation_system, perception_system):
    """Create a tool registry with robot-specific tools."""
    registry = ToolRegistry()

    # Navigation tools
    registry.register(
        name="navigate_to",
        description="Navigate the robot to a specific location. "
                   "Use this when you need to move to a known location.",
        parameters={
            "location": {
                "type": "string",
                "description": "Name or description of the destination location"
            },
            "position": {
                "type": "object",
                "description": "Optional XYZ position if location name is not known",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate"},
                    "y": {"type": "number", "description": "Y coordinate"},
                    "z": {"type": "number", "description": "Z coordinate"}
                }
            }
        },
        func=lambda location, position=None: navigation_system.navigate(
            location=location, position=position
        )
    )

    # Manipulation tools
    registry.register(
        name="pick_up",
        description="Pick up an object. The robot must be near the object.",
        parameters={
            "object_name": {
                "type": "string",
                "description": "Name of the object to pick up"
            },
            "object_id": {
                "type": "string",
                "description": "Optional specific object ID if known"
            }
        },
        func=lambda object_name, object_id=None: manipulation_system.pick_up(
            object_name, object_id
        )
    )

    registry.register(
        name="place_down",
        description="Place the currently held object at a location.",
        parameters={
            "location": {
                "type": "string",
                "description": "Where to place the object"
            }
        },
        func=lambda location: manipulation_system.place_down(location)
    )

    # Perception tools
    registry.register(
        name="scan_environment",
        description="Scan the environment to detect and locate objects.",
        parameters={
            "object_category": {
                "type": "string",
                "description": "Optional category to filter (e.g., 'cups', 'tools')"
            }
        },
        func=lambda object_category=None: perception_system.scan(
            object_category=object_category
        )
    )

    registry.register(
        name="locate_object",
        description="Search for a specific object by name.",
        parameters={
            "object_name": {
                "type": "string",
                "description": "Name or description of the object to find"
            }
        },
        func=lambda object_name: perception_system.locate(object_name)
    )

    # Information tools
    registry.register(
        name="get_state",
        description="Get the current robot state including position and held objects.",
        parameters={},
        func=lambda: navigation_system.get_state()
    )

    registry.register(
        name="check_battery",
        description="Check the current battery level.",
        parameters={},
        func=lambda: {"battery": navigation_system.get_battery_level()}
    )

    return registry
```

### Chain-of-Thought Reasoning

Chain-of-thought reasoning enables LLMs to solve complex problems by generating explicit reasoning steps rather than jumping directly to conclusions. This approach significantly improves performance on tasks requiring multi-step reasoning and enables the LLM to catch and correct its own mistakes.

```python
class ChainOfThoughtPlanner:
    """
    LLM planner using chain-of-thought reasoning.
    """

    def __init__(self, llm_client, tools: ToolRegistry):
        self.llm = llm_client
        self.tools = tools

    def plan_with_cot(self, goal: str, max_steps: int = 10) -> Dict:
        """
        Generate a plan using chain-of-thought reasoning.

        Args:
            goal: The task goal
            max_steps: Maximum reasoning steps

        Returns:
            Dict with reasoning trace and final plan
        """
        reasoning_trace = []
        current_state = self._get_initial_state()

        for step in range(max_steps):
            # Generate next reasoning step
            reasoning_step = self._generate_reasoning_step(
                goal, current_state, reasoning_trace
            )
            reasoning_trace.append(reasoning_step)

            # Check if we reached a conclusion
            if reasoning_step["type"] == "conclusion":
                return {
                    "reasoning": reasoning_trace,
                    "plan": reasoning_step["plan"],
                    "complete": True
                }

            # Check if we need to take action
            if reasoning_step["type"] == "action":
                action_result = self._execute_action(reasoning_step["action"])
                current_state = self._update_state(current_state, action_result)

        return {
            "reasoning": reasoning_trace,
            "plan": None,
            "complete": False,
            "message": "Maximum steps reached"
        }

    def _get_initial_state(self) -> Dict:
        """Get current robot state."""
        # This would call actual robot APIs
        return {
            "position": {"x": 0, "y": 0, "z": 0},
            "held_object": None,
            "battery": 85.0,
            "visible_objects": []
        }

    def _generate_reasoning_step(self, goal: str, state: Dict,
                                  trace: List) -> Dict:
        """Generate the next reasoning step."""
        # Build context from history
        history = "\n".join([
            f"Step {i+1}: {s['reasoning']}"
            for i, s in enumerate(trace)
        ]) if trace else "No steps taken yet."

        state_info = f"""
Current state:
- Position: {state['position']}
- Holding: {state['held_object'] or 'nothing'}
- Battery: {state['battery']}%
- Visible objects: {', '.join(state['visible_objects']) or 'none'}
"""

        prompt = f"""You are planning actions for a robot to achieve: "{goal}"

{state_info}

Previous reasoning steps:
{history}

Think step by step about what to do next. Consider:
1. What is the current situation?
2. What needs to be done to achieve the goal?
3. What information do you need?
4. What action should be taken?

Respond in this format:
### Reasoning
<Brief reasoning about current situation and next steps>

### Type
<action | question | conclusion>

### Action (if type is action)
<action_name>
Arguments: <JSON object>

### Question (if type is question)
<What you need to know>

### Plan (if type is conclusion)
<Summary of the complete plan as JSON>
"""

        response = self.llm.generate(prompt, max_tokens=1024)
        return self._parse_reasoning_response(response)

    def _parse_reasoning_response(self, response: str) -> Dict:
        """Parse LLM response into structured format."""
        result = {"type": "reasoning", "reasoning": ""}

        # Extract reasoning
        reasoning_match = re.search(
            r'### Reasoning\n(.*?)(?=\n###|\Z)',
            response, re.DOTALL
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        # Extract type
        type_match = re.search(r'### Type\n(\w+)', response)
        if type_match:
            result["type"] = type_match.group(1).lower()

        # Extract action
        if result["type"] == "action":
            action_match = re.search(
                r'### Action\n(\w+)\nArguments: ({.*?})',
                response, re.DOTALL
            )
            if action_match:
                result["action"] = {
                    "name": action_match.group(1),
                    "arguments": json.loads(action_match.group(2))
                }

        # Extract question
        elif result["type"] == "question":
            question_match = re.search(
                r'### Question\n(.*?)$',
                response, re.DOTALL
            )
            if question_match:
                result["question"] = question_match.group(1).strip()

        # Extract plan
        elif result["type"] == "conclusion":
            plan_match = re.search(
                r'### Plan\n(.*?)$',
                response, re.DOTALL
            )
            if plan_match:
                result["plan"] = json.loads(plan_match.group(1))

        return result

    def _execute_action(self, action: Dict) -> Dict:
        """Execute an action and return result."""
        try:
            result = self.tools.execute(action["name"], action["arguments"])
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_state(self, state: Dict, action_result: Dict) -> Dict:
        """Update state based on action result."""
        # This would update based on actual action effects
        new_state = state.copy()
        if action_result.get("success"):
            # Update state based on action result
            pass
        return new_state
```

## Memory and Context Management

Effective LLM-based planning requires careful management of context and memory. The context window limits how much information can be provided to the LLM, while working and long-term memory enable the system to maintain coherent reasoning across extended interactions.

### Working Memory System

Working memory maintains the current context for the LLM, including the current plan, recent observations, and active goals. This memory must be carefully managed to stay within context limits while preserving essential information.

```python
from collections import deque
from dataclasses import dataclass

@dataclass
class MemoryItem:
    """An item in the working memory."""
    content: str
    timestamp: datetime
    importance: float  # 0-1 scale
    category: str  # 'observation', 'plan', 'goal', 'constraint'

class WorkingMemory:
    """
    Manages working memory for the cognitive planner.
    """

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.items = []
        self.token_counts = {}  # Track tokens per category

    def add(self, content: str, category: str, importance: float = 0.5):
        """Add an item to working memory."""
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            category=category
        )
        self.items.append(item)
        self._trim_if_needed()

    def get_context(self, categories: List[str] = None) -> str:
        """Get concatenated context from memory."""
        if categories is None:
            categories = ['observation', 'plan', 'goal', 'constraint']

        relevant = [item for item in self.items
                   if item.category in categories]

        # Sort by importance
        relevant.sort(key=lambda x: x.importance, reverse=True)

        return "\n".join([item.content for item in relevant])

    def _trim_if_needed(self):
        """Remove low-importance items if over token limit."""
        # Simplified token counting
        current_tokens = sum(len(item.content.split()) for item in self.items)

        while current_tokens > self.max_tokens / 4:  # Use 25% of limit for memory
            # Remove lowest importance non-essential item
            removable = [i for i, item in enumerate(self.items)
                        if item.category != 'goal']  # Never remove goals
            if not removable:
                break  # Can't trim more

            to_remove = min(removable, key=lambda i: self.items[i].importance)
            current_tokens -= len(self.items[to_remove].content.split())
            del self.items[to_remove]

class LongTermMemory:
    """
    Stores and retrieves long-term memories.
    """

    def __init__(self):
        self.episodic_memory = []  # Experiences
        self.semantic_memory = {}  # Knowledge
        self.procedural_memory = {}  # Skills

    def store_experience(self, experience: Dict):
        """Store an episodic experience."""
        self.episodic_memory.append({
            **experience,
            "timestamp": datetime.now()
        })

    def store_knowledge(self, key: str, knowledge: str):
        """Store semantic knowledge."""
        self.semantic_memory[key] = knowledge

    def retrieve_relevant(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant memories using embedding similarity."""
        # In practice, would use embedding similarity
        # Simplified version returns recent relevant memories
        relevant = []

        for exp in reversed(self.episodic_memory):
            if query.lower() in exp.get("description", "").lower():
                relevant.append(exp)
                if len(relevant) >= n_results:
                    break

        return relevant
```

## Safety and Reliability

LLM-based planning systems must be designed with safety in mind, as incorrect or unsafe plans could damage the robot or its environment. Several strategies mitigate these risks.

### Plan Validation and Safety Checking

Before execution, plans should be validated against safety constraints and checked for feasibility. This includes verifying that each action is safe to perform, checking that prerequisites are met, and ensuring the overall plan is coherent.

```python
class SafetyValidator:
    """
    Validates plans for safety and feasibility.
    """

    def __init__(self, robot_constraints: Dict, environment_model):
        self.constraints = robot_constraints
        self.env = environment_model

        # Define safety rules
        self.safety_rules = [
            self._check_battery_safety,
            self._check_collision_safety,
            self._check_force_limits,
            self._check_speed_limits
        ]

    def validate_plan(self, plan: Plan, current_state: Dict) -> Dict:
        """
        Validate a complete plan.

        Returns:
            Dict with 'valid', 'warnings', 'errors', and 'suggestions'
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }

        # Check each action
        for task in plan.tasks:
            task_result = self._validate_task(task, current_state)
            result["warnings"].extend(task_result["warnings"])
            result["errors"].extend(task_result["errors"])

            # Update state for next task
            current_state = self._simulate_state_change(current_state, task)

        # Check plan coherence
        coherence_issues = self._check_plan_coherence(plan)
        result["warnings"].extend(coherence_issues)

        result["valid"] = len(result["errors"]) == 0
        return result

    def _validate_task(self, task: Task, state: Dict) -> Dict:
        """Validate a single task."""
        result = {"warnings": [], "errors": []}

        # Check prerequisites
        if task.dependencies:
            incomplete = [d for d in task.dependencies
                         if not self._is_dependency_met(d, state)]
            if incomplete:
                result["errors"].append(
                    f"Task {task.id} has unmet dependencies: {incomplete}"
                )

        # Check battery
        battery_warning = self._check_battery_safety(task, state)
        if battery_warning:
            result["warnings"].append(battery_warning)

        # Check action-specific constraints
        if task.action_type in self.constraints:
            for constraint_check in self.constraints[task.action_type]:
                if not constraint_check(task, state):
                    result["errors"].append(
                        f"Action {task.action_type} violates constraint"
                    )

        return result

    def _check_battery_safety(self, task: Task, state: Dict) -> Optional[str]:
        """Check if battery is sufficient for task."""
        battery = state.get("battery", 100)
        estimated_cost = self._estimate_battery_cost(task)

        if battery < estimated_cost + 20:  # Keep 20% buffer
            return f"Battery low ({battery}%), task requires ~{estimated_cost}%"
        return None

    def _estimate_battery_cost(self, task: Task) -> float:
        """Estimate battery cost for a task."""
        cost_estimates = {
            "navigate_to": 5.0,
            "pick_up": 3.0,
            "place_down": 2.0,
            "scan_environment": 1.0
        }
        return cost_estimates.get(task.action_type, 5.0)

    def _is_dependency_met(self, dep_id: str, state: Dict) -> bool:
        """Check if a dependency is met."""
        # Simplified check
        return True

    def _simulate_state_change(self, state: Dict, task: Task) -> Dict:
        """Simulate how state changes after task."""
        new_state = state.copy()

        if task.action_type == "pick_up":
            new_state["held_object"] = task.parameters.get("object_name")

        elif task.action_type == "place_down":
            new_state["held_object"] = None

        return new_state

    def _check_plan_coherence(self, plan: Plan) -> List[str]:
        """Check if the plan is coherent and sensible."""
        issues = []

        # Check for duplicate tasks
        actions = [t.action_type for t in plan.tasks]
        if len(actions) != len(set(actions)):
            issues.append("Plan may contain duplicate actions")

        # Check for impossible sequences
        for i in range(len(plan.tasks) - 1):
            if (plan.tasks[i].action_type == "place_down" and
                plan.tasks[i + 1].action_type == "pick_up"):
                issues.append(
                    "Warning: Placing object before picking anything up"
                )

        return issues
```

## Key Takeaways

LLM-based cognitive planning enables robots to perform complex reasoning and task decomposition that was previously impossible with traditional planning approaches. By leveraging the world knowledge and reasoning capabilities of LLMs, robots can understand natural language instructions, adapt to novel situations, and generate appropriate action sequences.

- **LLM foundations** include world knowledge, chain-of-thought reasoning, and tool use
- **Cognitive architecture** includes perception, memory, reasoning, and execution layers
- **Tool calling** enables LLMs to interact with the physical world through structured APIs
- **Chain-of-thought** improves complex reasoning through explicit step-by-step thinking
- **Memory management** maintains context across extended interactions
- **Safety validation** ensures generated plans are safe and feasible
- **Practical integration** requires careful prompt engineering and error handling

With cognitive planning established, we can now explore humanoid locomotion—the fundamental capability that enables humanoid robots to move through human environments.
