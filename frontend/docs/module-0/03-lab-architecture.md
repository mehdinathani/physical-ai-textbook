---
sidebar_position: 3
---

# Lab Architecture & Safety

Building humanoid robots requires more than theoretical knowledge—it demands carefully designed physical spaces where mechanical, electrical, and software systems can be developed, tested, and refined safely. This chapter explores the architecture of effective robotics laboratories, the safety protocols that protect both humans and equipment, and the infrastructure considerations that enable productive research and development.

## Physical Space Requirements

A robotics laboratory for humanoid development must accommodate multiple interconnected systems: the robots themselves, their power and computing infrastructure, debugging and monitoring equipment, and the safety systems that prevent accidents. The space must also facilitate collaboration among mechanical engineers, electrical engineers, software developers, and researchers—all working on systems that can move unpredictably at high speed and force.

### Space Planning Principles

Humanoid robots require more space than their physical dimensions suggest. A 1.7-meter humanoid with arms fully extended occupies a cylinder approximately 2 meters in diameter. When that robot falls—or is deliberately pushed to test recovery—the effective workspace expands dramatically. Planning for a single humanoid workcell requires at least 4×4 meters of clear floor space, with 6×6 meters being preferable.

The ceiling height must accommodate the robot's full reach plus any equipment mounted above. For full-sized humanoids, ceilings should be at least 3 meters high, with 4 meters preferred to allow for overhead cameras, lighting rigs, and emergency stop mechanisms.

Flooring must support the concentrated load of the robot plus its stand or gantry system. Industrial epoxy coating or raised access panels provide durable, easy-to-clean surfaces that can handle occasional impacts from fallen equipment. Anti-fatigue mats in standing work areas reduce strain during long debugging sessions.

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class WorkcellDimensions:
    """Defines the physical dimensions of a robot workcell."""
    width_m: float = 4.0
    height_m: float = 3.0
    depth_m: float = 4.0
    clear_radius_m: float = 1.5  # Radius of falling robot envelope
    overhead_clearance_m: float = 0.5

    @property
    def floor_area_m2(self) -> float:
        return self.width_m * self.depth_m

    @property
    def volume_m3(self) -> float:
        return self.floor_area_m2 * self.height_m

    @property
    def effective_workspace_diameter(self) -> float:
        return 2 * self.clear_radius_m + 0.5  # Robot width + margin

@dataclass
class LoadSpecification:
    """Defines load requirements for floor and mounting points."""
    point_load_kg: float = 200  # Maximum point load (robot foot)
    distributed_load_kg_m2: float = 500  # Distributed load capacity
    overhead_capacity_kg: float = 50  # Overhead equipment capacity

    def check_floor_rating(self, robot_mass_kg: float,
                          foot_area_m2: float) -> Tuple[bool, str]:
        """Check if floor can support the robot."""
        point_pressure = robot_mass_kg / foot_area_m2

        if point_pressure > self.distributed_load_kg_m2 * 1000:  # Convert to Pa
            return False, f"Point pressure {point_pressure:.0f} Pa exceeds rating"

        return True, "Floor rating OK"

class LabPlanner:
    """
    Plans robotics laboratory layout and infrastructure.
    """

    def __init__(self, num_robots: int = 2):
        self.num_robots = num_robots
        self.workcells: List[WorkcellDimensions] = []
        self.equipment_locations: Dict[str, Tuple[float, float, float]] = {}

    def add_workcell(self, dimensions: WorkcellDimensions) -> int:
        """Add a workcell and return its index."""
        self.workcells.append(dimensions)
        return len(self.workcells) - 1

    def calculate_space_requirements(self) -> Dict:
        """Calculate total space requirements for the lab."""
        if not self.workcells:
            return {'error': 'No workcells defined'}

        # Each workcell needs surrounding space for access
        access_margin = 1.0  # meters around each workcell

        total_width = sum(w.width_m + access_margin for w in self.workcells)
        total_depth = max(w.depth_m for w in self.workcells) + access_margin * 2
        total_height = max(w.height_m for w in self.workcells)

        return {
            'total_floor_area_m2': total_width * total_depth,
            'total_volume_m3': total_width * total_depth * total_height,
            'recommended_ceiling_height_m': total_height + 0.5,
            'minimum_door_width_m': max(w.width_m for w in self.workcells) * 0.8,
            'recommended_door_width_m': max(w.width_m for w in self.workcells) + 0.5
        }

    def calculate_power_requirements(self, compute_power_watts: float = 2000,
                                      actuation_power_watts: float = 5000,
                                      lighting_power_watts: float = 500) -> Dict:
        """Calculate electrical power requirements."""
        robot_power = (compute_power_watts + actuation_power_watts) * self.num_robots

        # Add 30% margin for overhead
        total_power = (robot_power + lighting_power_watts) * 1.3

        # Convert to appropriate units
        return {
            'robot_compute_watts': compute_power_watts * self.num_robots,
            'robot_actuation_watts': actuation_power_watts * self.num_robots,
            'lighting_watts': lighting_power_watts,
            'total_watts': total_power,
            'total_kva': total_power / 1000 * 1.2,  # Power factor adjustment
            'recommended_circuit_amps_240v': total_power / 240 / 0.8  # 80% derating
        }

    def generate_layout_recommendations(self) -> List[str]:
        """Generate recommendations for lab layout."""
        recommendations = []

        # Zone recommendations
        recommendations.append("Zone 1: Robot workcells (clear floor, high ceiling)")
        recommendations.append("Zone 2: Development workstations (near test areas)")
        recommendations.append("Zone 3: Hardware workshop (separated from test areas)")
        recommendations.append("Zone 4: Server and networking (climate controlled)")

        # Safety recommendations
        recommendations.append("Install E-stops within 2m of all workcells")
        recommendations.append("Maintain 1m clearance around each workcell")
        recommendations.append("Use non-slip, impact-absorbing flooring")

        # Infrastructure recommendations
        recommendations.append("Dedicated 240V circuits for robot charging")
        recommendations.append("Network drops at each workstation and workcell")
        recommendations.append("Overhead camera mounts at 3m height minimum")

        return recommendations

# Example lab planning
planner = LabPlanner(num_robots=2)

# Add workcells
planner.add_workcell(WorkcellDimensions(
    width_m=5.0, depth_m=5.0, height_m=3.5, clear_radius_m=1.8
))
planner.add_workcell(WorkcellDimensions(
    width_m=4.0, depth_m=4.0, height_m=3.5, clear_radius_m=1.5
))

print("Laboratory Space Analysis")
print("=" * 50)
space = planner.calculate_space_requirements()
for key, value in space.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.1f}")
    else:
        print(f"  {key}: {value}")

print("\nPower Requirements:")
power = planner.calculate_power_requirements()
for key, value in power.items():
    print(f"  {key}: {value:.1f}")

print("\nLayout Recommendations:")
for rec in planner.generate_layout_recommendations()[:5]:
    print(f"  - {rec}")
```

### Zoning and Layout

Effective laboratories divide into functional zones that balance accessibility with safety:

**Development Zones** contain workstations for programming, simulation, and code development. These areas should have clear sightlines to the test arena but physical separation to protect personnel during robot operation. Dual-monitor workstations with wireless connectivity allow developers to iterate quickly without being tethered to the robot.

**Test Arenas** are the controlled spaces where robots operate. These areas need safety perimeters, impact-absorbing surfaces, and clear emergency access paths. Lighting should be even and shadow-free to support computer vision. Multiple camera angles—from RGB, depth, and IR sources—enable comprehensive observation and recording.

**Hardware Workshop** areas support mechanical and electrical work: 3D printing for custom fixtures, electronics workbenches with ESD protection, basic machining capability, and storage for spare parts and tools. This zone should be physically separated from active test areas.

**Server and Networking Infrastructure** houses the simulation servers, network equipment, and data storage systems. This space needs proper cooling, vibration isolation (to prevent disruption to sensitive calibration equipment), and physical security.

## Power Infrastructure

Humanoid robots consume substantial electrical power, and providing that power safely and reliably requires careful infrastructure design.

### Power Distribution

A typical humanoid robot operates from a 24-48V battery system with peak current draws of 50-100A during aggressive motion. The laboratory must provide:

**High-Current Outlets**: Dedicated 30A or 60A circuits on 240V feeds support battery charging and direct power supply operation. These circuits require appropriate circuit breakers, cable gauges, and outlet types. NEMA 6-50 or L6-30 outlets are common choices for industrial robot installations.

**Uninterruptible Power Supply (UPS)**: Critical infrastructure—servers, network equipment, safety systems—should run on UPS power to prevent data loss and enable graceful shutdown during grid fluctuations or brief outages.

**Power Monitoring**: Real-time monitoring of current, voltage, and power consumption helps identify problems before they cause failures. Many laboratory incidents begin with subtle power quality issues that escalate into dangerous situations.

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class PowerPhase(Enum):
    SINGLE_PHASE = "single_phase"
    THREE_PHASE = "three_phase"

@dataclass
class CircuitSpec:
    """Specification for an electrical circuit."""
    name: str
    voltage: float  # Volts
    current_amps: float
    phase: PowerPhase
    purpose: str  # e.g., "robot_charging", "workstation", "lighting"

@dataclass
class PowerReading:
    """Real-time power reading."""
    voltage: float
    current: float
    power_factor: float
    timestamp: float

class PowerDistributionSystem:
    """
    Models and manages laboratory power distribution.
    """

    def __init__(self, total_capacity_va: float = 50000):
        self.total_capacity = total_capacity_va
        self.circuits: List[CircuitSpec] = []
        self.current_loads: Dict[str, float] = {}  # Current per circuit
        self.readings: List[PowerReading] = []
        self.safety_margin = 0.8  # 80% of rated capacity max

    def add_circuit(self, circuit: CircuitSpec):
        """Add a circuit to the distribution system."""
        self.circuits.append(circuit)
        self.current_loads[circuit.name] = 0.0

    def calculate_available_power(self, circuit_name: Optional[str] = None) -> float:
        """Calculate available power capacity."""
        if circuit_name:
            circuit = next((c for c in self.circuits if c.name == circuit_name), None)
            if circuit:
                used = self.current_loads.get(circuit_name, 0)
                available = circuit.voltage * circuit.current_amps * self.safety_margin - used
                return max(0, available)
            return 0

        total_available = self.total_capacity * self.safety_margin
        for name, load in self.current_loads.items():
            circuit = next((c for c in self.circuits if c.name == name), None)
            if circuit:
                total_available -= load * circuit.voltage

        return max(0, total_available)

    def add_load(self, circuit_name: str, load_watts: float) -> bool:
        """Add a load to a circuit. Returns True if successful."""
        circuit = next((c for c in self.circuits if c.name == circuit_name), None)
        if not circuit:
            return False

        new_load = self.current_loads.get(circuit_name, 0) + load_watts
        max_load = circuit.voltage * circuit.current_amps * self.safety_margin

        if new_load > max_load:
            return False  # Would exceed capacity

        self.current_loads[circuit_name] = new_load
        return True

    def simulate_fault(self, circuit_name: str, fault_type: str) -> Dict:
        """Simulate a fault condition on a circuit."""
        circuit = next((c for c in self.circuits if c.name == circuit_name), None)
        if not circuit:
            return {'error': 'Circuit not found'}

        fault_responses = {
            'overcurrent': {
                'trip_current_ratio': 1.2,  # Trip at 120% of rated
                'trip_time_ms': 100,
                'protective_device': 'Circuit breaker'
            },
            'short_circuit': {
                'trip_current_ratio': 10.0,
                'trip_time_ms': 10,
                'protective_device': 'Circuit breaker'
            },
            'ground_fault': {
                'trip_current_mA': 30,
                'trip_time_ms': 50,
                'protective_device': 'GFCI'
            }
        }

        return fault_responses.get(fault_type, {'error': 'Unknown fault type'})

    def generate_sizing_report(self) -> Dict:
        """Generate power sizing report."""
        total_robot_power = 0
        total_workstation_power = 0
        total_lighting_power = 0

        for circuit in self.circuits:
            if 'robot' in circuit.purpose.lower():
                total_robot_power += circuit.voltage * circuit.current_amps
            elif 'workstation' in circuit.purpose.lower():
                total_workstation_power += circuit.voltage * circuit.current_amps
            elif 'light' in circuit.purpose.lower():
                total_lighting_power += circuit.voltage * circuit.current_amps

        return {
            'total_capacity_va': self.total_capacity,
            'robot_power_va': total_robot_power,
            'workstation_power_va': total_workstation_power,
            'lighting_power_va': total_lighting_power,
            'headroom_va': self.calculate_available_power(),
            'recommendation': self._get_sizing_recommendation()
        }

    def _get_sizing_recommendation(self) -> str:
        """Get recommendations for power sizing."""
        headroom = self.calculate_available_power()
        if headroom < self.total_capacity * 0.2:
            return "WARNING: Less than 20% headroom. Consider adding circuits."
        return "Power capacity adequate with good headroom."

# Example power distribution sizing
pds = PowerDistributionSystem(total_capacity_va=100000)

# Add circuits
pds.add_circuit(CircuitSpec(
    name="robot_1_charging", voltage=240, current_amps=30,
    phase=PowerPhase.SINGLE_PHASE, purpose="robot_charging"
))
pds.add_circuit(CircuitSpec(
    name="robot_2_charging", voltage=240, current_amps=30,
    phase=PowerPhase.SINGLE_PHASE, purpose="robot_charging"
))
pds.add_circuit(CircuitSpec(
    name="workstations", voltage=120, current_amps=20,
    phase=PowerPhase.SINGLE_PHASE, purpose="workstation"
))
pds.add_circuit(CircuitSpec(
    name="lighting", voltage=120, current_amps=15,
    phase=PowerPhase.SINGLE_PHASE, purpose="lighting"
))

# Add loads
pds.add_load("robot_1_charging", 4000)  # Charging at ~17A
pds.add_load("robot_2_charging", 0)     # Not charging
pds.add_load("workstations", 1500)      # Developer workstations
pds.add_load("lighting", 500)           # Partial lighting

print("Power Distribution Analysis")
print("=" * 50)
report = pds.generate_sizing_report()
for key, value in report.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.0f} VA")
    else:
        print(f"  {key}: {value}")
```

### Grounding and Electrical Safety

Proper grounding protects both personnel and equipment. All conductive surfaces in the test area should be bonded to a common ground point. The robot's chassis should have a dedicated ground connection that doesn't depend on power supply cables, which can fail during robot movement.

Isolation transformers or Ground Fault Circuit Interrupters (GFCIs) provide additional protection against electric shock. In humid environments or where condensation is possible, these protections are essential.

## Computing Infrastructure

Modern robotics development requires substantial computing resources distributed across on-robot, on-premise, and cloud systems.

### Network Architecture

A dedicated laboratory network, physically or virtually separated from building infrastructure, provides reliable, low-latency communication. This network typically includes:

**Wired Backbone**: 10 Gigabit or faster connections between servers, switches, and the robot's docking station. Wired connections are essential for high-bandwidth operations like transferring camera data during debugging sessions.

**Wireless Access Points**: High-bandwidth wireless coverage throughout the test area supports mobile operation and development workflows. Wi-Fi 6E or Wi-Fi 7 access points provide the low latency and high throughput necessary for wireless teleoperation.

**Network Segmentation**: Separate VLANs for development, production robots, and infrastructure prevent accidental interference and contain potential security breaches. Critical systems should be air-gapped from external networks.

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class NetworkZone(Enum):
    MANAGEMENT = "management"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    INFRASTRUCTURE = "infrastructure"

@dataclass
class NetworkDevice:
    """Represents a network device in the lab."""
    name: str
    ip_address: str
    mac_address: str
    zone: NetworkZone
    bandwidth_mbps: float
    latency_ms: float

@dataclass
class NetworkLink:
    """Represents a network link between devices."""
    device_a: str
    device_b: str
    bandwidth_mbps: float
    latency_ms: float
    link_type: str  # "wired", "wireless"

class NetworkArchitecture:
    """
    Designs and analyzes laboratory network architecture.
    """

    def __init__(self):
        self.devices: List[NetworkDevice] = []
        self.links: List[NetworkLink] = []
        self.vlans: Dict[str, List[str]] = {}  # VLAN name -> device names

    def add_device(self, device: NetworkDevice):
        """Add a network device."""
        self.devices.append(device)

    def add_link(self, link: NetworkLink):
        """Add a network link."""
        self.links.append(link)

    def configure_vlan(self, vlan_name: str, device_names: List[str]):
        """Configure VLAN assignment."""
        self.vlans[vlan_name] = device_names

    def calculate_bandwidth_requirements(self) -> Dict:
        """Calculate network bandwidth requirements."""
        requirements = {
            'robot_telemetry': 0,
            'camera_streams': 0,
            'point_cloud': 0,
            'development': 0,
            'infrastructure': 0,
            'total': 0
        }

        for device in self.devices:
            if 'robot' in device.name.lower():
                requirements['robot_telemetry'] += device.bandwidth_mbps
            elif 'camera' in device.name.lower():
                requirements['camera_streams'] += device.bandwidth_mbps
            elif 'depth' in device.name.lower():
                requirements['point_cloud'] += device.bandwidth_mbps
            elif 'workstation' in device.name.lower() or 'dev' in device.name.lower():
                requirements['development'] += device.bandwidth_mbps
            else:
                requirements['infrastructure'] += device.bandwidth_mbps

        requirements['total'] = sum(v for k, v in requirements.items() if k != 'total')

        return requirements

    def check_latency_budget(self, device_a: str, device_b: str,
                             max_latency_ms: float = 10) -> Dict:
        """Check if latency between two devices meets budget."""
        # Find path (simplified - just direct link)
        link = next((l for l in self.links if
                    (l.device_a == device_a and l.device_b == device_b) or
                    (l.device_a == device_b and l.device_b == device_a)), None)

        if not link:
            return {'path_found': False, 'latency_ms': None, 'meets_budget': None}

        return {
            'path_found': True,
            'latency_ms': link.latency_ms,
            'meets_budget': link.latency_ms <= max_latency_ms
        }

    def generate_architecture_report(self) -> Dict:
        """Generate network architecture report."""
        bandwidth = self.calculate_bandwidth_requirements()

        # Count devices per zone
        zone_counts = {}
        for device in self.devices:
            zone_counts[device.zone.value] = zone_counts.get(device.zone.value, 0) + 1

        # Calculate network redundancy
        connected_pairs = set()
        for link in self.links:
            connected_pairs.add(frozenset([link.device_a, link.device_b]))

        return {
            'total_devices': len(self.devices),
            'total_links': len(self.links),
            'devices_per_zone': zone_counts,
            'bandwidth_requirements_mbps': bandwidth,
            'recommended_switch_capacity': bandwidth['total'] * 1.5,  # 50% headroom
            'vlan_count': len(self.vlans)
        }

# Example network architecture
net = NetworkArchitecture()

# Add devices
net.add_device(NetworkDevice(
    name="robot_1", ip_address="10.0.1.10", mac_address="00:11:22:33:44:55",
    zone=NetworkZone.PRODUCTION, bandwidth_mbps=1000, latency_ms=1
))
net.add_device(NetworkDevice(
    name="robot_2", ip_address="10.0.1.11", mac_address="00:11:22:33:44:56",
    zone=NetworkZone.PRODUCTION, bandwidth_mbps=1000, latency_ms=1
))
net.add_device(NetworkDevice(
    name="dev_workstation_1", ip_address="10.0.2.10", mac_address="AA:BB:CC:DD:EE:01",
    zone=NetworkZone.DEVELOPMENT, bandwidth_mbps=10000, latency_ms=0.5
))
net.add_device(NetworkDevice(
    name="dev_workstation_2", ip_address="10.0.2.11", mac_address="AA:BB:CC:DD:EE:02",
    zone=NetworkZone.DEVELOPMENT, bandwidth_mbps=10000, latency_ms=0.5
))
net.add_device(NetworkDevice(
    name="sim_server", ip_address="10.0.3.10", mac_address="BB:CC:DD:EE:FF:01",
    zone=NetworkZone.INFRASTRUCTURE, bandwidth_mbps=40000, latency_ms=0.2
))

# Add links
net.add_link(NetworkLink(
    device_a="robot_1", device_b="dev_workstation_1",
    bandwidth_mbps=1000, latency_ms=1, link_type="wireless"
))
net.add_link(NetworkLink(
    device_a="robot_2", device_b="dev_workstation_2",
    bandwidth_mbps=1000, latency_ms=1, link_type="wireless"
))
net.add_link(NetworkLink(
    device_a="dev_workstation_1", device_b="sim_server",
    bandwidth_mbps=10000, latency_ms=0.5, link_type="wired"
))

# Configure VLANs
net.configure_vlan("production", ["robot_1", "robot_2"])
net.configure_vlan("development", ["dev_workstation_1", "dev_workstation_2"])

print("Network Architecture Analysis")
print("=" * 50)
report = net.generate_architecture_report()
for key, value in report.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {key}: {value}")
```

### Data Management

Robot development generates enormous quantities of data: camera streams, LiDAR point clouds, IMU measurements, motor currents, and the resulting motion trajectories. A comprehensive data management strategy includes:

**Recording Infrastructure**: High-speed NVMe SSDs capture raw sensor data during test sessions. A typical 30-minute session with multiple cameras and full-state recording can generate 50-100GB of data. Plan storage capacity accordingly.

**Data Organization**: Consistent naming conventions, automatic metadata tagging, and searchable databases make it possible to find and analyze historical data. The cost of good data management is far less than the cost of searching through unorganized terabytes.

**Backup and Archival**: Critical data should follow a 3-2-1 backup strategy: three copies on two different media types with one copy offsite. Cloud storage services provide convenient offsite backup for non-sensitive data.

## Safety Systems

The potential hazards in a robotics laboratory require multiple layers of protection. No single safety mechanism is sufficient; defense in depth is essential.

### Emergency Stop Systems

Emergency stop (E-stop) buttons must be positioned throughout the laboratory such that any operator can reach one within two steps. The E-stop system should:

**Cut All Power**: The E-stop circuit must directly interrupt power to actuators, not merely signal software to stop. Software can fail; hardware interconnects cannot.

**Latch in Safe State**: Once activated, E-stops must require manual reset before the system can restart. This prevents automatic restart after a brief power blip.

**Override All Other Controls**: The E-stop circuit must have highest priority, disconnected from any other system that might delay its operation.

**Provide Clear Status Indication**: Both activated and normal states should be clearly visible, with indicators that remain visible even if the robot falls and blocks line of sight.

```python
import time
from dataclasses import dataclass, field
from typing import List, Callable, Optional
from enum import Enum
import threading

class EStopState(Enum):
    NOT_PRESSED = "not_pressed"
    PRESSED = "pressed"
    LATCHED = "latched"
    RESETTING = "resetting"

@dataclass
class EStopButton:
    """Represents an E-stop button."""
    id: str
    location: str
    normally_closed: bool = True  # NC contacts for fail-safe

class SafetyMonitor:
    """
    Central safety monitoring system for the laboratory.
    """

    def __init__(self):
        self.estop_buttons: List[EStopButton] = []
        self.estop_state = EStopState.NOT_PRESSED
        self.power_relays: List[str] = []  # Names of power relays
        self.alarm_callbacks: List[Callable] = []
        self.status_listeners: List[Callable] = []
        self._lock = threading.Lock()

    def add_estop_button(self, button: EStopButton):
        """Add an E-stop button to the system."""
        self.estop_buttons.append(button)

    def register_alarm_callback(self, callback: Callable[[str], None]):
        """Register callback for alarm conditions."""
        self.alarm_callbacks.append(callback)

    def register_status_callback(self, callback: Callable[[str], None]):
        """Register callback for status changes."""
        self.status_listeners.append(callback)

    def trigger_estop(self, button_id: Optional[str] = None) -> bool:
        """
        Trigger emergency stop.
        If button_id is None, simulates general E-stop activation.
        """
        with self._lock:
            if self.estop_state == EStopState.PRESSED:
                return False  # Already pressed

            self.estop_state = EStopState.PRESSED

            # Cut power to actuators
            self._cut_power()

            # Notify listeners
            source = button_id if button_id else "central"
            for callback in self.alarm_callbacks:
                callback(f"EMERGENCY STOP ACTIVATED by {source}")

            # Latch the E-stop
            self.estop_state = EStopState.LATCHED

            return True

    def reset_estop(self) -> bool:
        """
        Reset emergency stop system.
        Requires manual intervention to clear latch.
        """
        with self._lock:
            if self.estop_state != EStopState.LATCHED:
                return False

            self.estop_state = EStopState.RESETTING

            # Verify all E-stop buttons are released
            for button in self.estop_buttons:
                if not self._check_button_released(button):
                    self.estop_state = EStopState.LATCHED
                    return False

            # Restore power gradually
            self._restore_power()

            self.estop_state = EStopState.NOT_PRESSED

            for callback in self.status_listeners:
                callback("E-STOP RESET - System operational")

            return True

    def check_safe_to_operate(self) -> Tuple[bool, str]:
        """Check if system is safe to operate."""
        with self._lock:
            if self.estop_state != EStopState.NOT_PRESSED:
                return False, f"E-stop in state: {self.estop_state.value}"

            # Check all E-stop buttons
            for button in self.estop_buttons:
                if not self._check_button_released(button):
                    return False, f"Button {button.id} pressed"

            return True, "All systems nominal"

    def _check_button_released(self, button: EStopButton) -> bool:
        """Check if an E-stop button is released (not pressed)."""
        # In real implementation, read from hardware
        return True

    def _cut_power(self):
        """Cut power to actuators."""
        for relay in self.power_relays:
            self._set_relay(relay, False)

    def _restore_power(self):
        """Restore power to actuators."""
        for relay in self.power_relays:
            self._set_relay(relay, True)

    def _set_relay(self, relay: str, state: bool):
        """Set relay state (hardware abstraction)."""
        pass

# Example safety system setup
safety = SafetyMonitor()

# Add E-stop buttons
safety.add_estop_button(EStopButton("estop_1", "Test area - North wall"))
safety.add_estop_button(EStopButton("estop_2", "Test area - South wall"))
safety.add_estop_button(EStopButton("estop_3", "Development area"))

# Register callbacks
def alarm_handler(message: str):
    print(f"ALARM: {message}")

def status_handler(message: str):
    print(f"STATUS: {message}")

safety.register_alarm_callback(alarm_handler)
safety.register_status_callback(status_handler)

print("Safety System Test")
print("=" * 50)

# Check initial state
safe, msg = safety.check_safe_to_operate()
print(f"Initial state: {msg}")

# Trigger E-stop
print("\nTriggering E-stop...")
safety.trigger_estop("estop_1")

safe, msg = safety.check_safe_to_operate()
print(f"After E-stop: {msg}")

# Attempt to operate (should fail)
print(f"Safe to operate: {safe}")

# Reset E-stop
print("\nResetting E-stop...")
if safety.reset_estop():
    print("Reset successful")
else:
    print("Reset failed - check E-stop buttons")
```

### Physical Barriers

Physical barriers separate people from operating robots. For humanoid development, these barriers take several forms:

**Perimeter Fencing**: Chain-link or mesh fencing surrounds the test area with locked access gates. The fence should be tall enough to prevent climbing and positioned far enough from the robot that a falling robot cannot reach over it.

**Light Curtains and Mats**: Pressure-sensitive mats and infrared light curtains detect human presence in protected zones. These systems can trigger automatic stops or prevent robot activation when triggered.

**Cage Systems**: For smaller robots or aggressive testing, full cage enclosures provide complete isolation during operation. Access doors have interlock switches that prevent robot operation when open.

### Software Safety Limits

Software systems enforce limits that physical systems cannot easily provide:

**Velocity Limits**: Maximum velocities for all joints should be set conservatively, with wider margins during initial testing. These limits should be enforced at multiple levels: in the motor controller, in the ros_control hierarchy, and in high-level planning.

**Workspace Boundaries**: Virtual walls define the region where the robot is allowed to move. Trajectories that would exit this workspace are rejected before execution.

**Force Limits**: Maximum forces for end-effectors and contact points prevent damage to objects and people. These limits should be enforced through both trajectory planning and real-time monitoring.

**Watchdog Timers**: All critical systems must have watchdog timers that trigger safe shutdown if communications are interrupted. A robot that loses connection to its safety monitoring system should stop moving, not continue its last command.

## Environmental Considerations

The laboratory environment affects both robot performance and human comfort.

### Temperature and Humidity

Electronics generate heat, and high-performance computing systems generate substantial heat. The laboratory must dissipate this heat while maintaining conditions suitable for human occupancy:

**Cooling Capacity**: Plan for at least 3kW of heat generation per high-power workstation, plus similar amounts for active robot testing. Air conditioning must handle both the equipment heat and the heat from human occupants.

**Air Circulation**: Proper air circulation prevents hot spots and ensures consistent sensor readings. For IMU-equipped robots, air currents can affect measurements if sensors are not adequately isolated.

**Humidity Control**: Extreme humidity can cause condensation on electronics, leading to short circuits or corrosion. In humid climates, dehumidification may be necessary.

### Lighting

Consistent, even lighting supports both human operators and computer vision systems:

**General Illumination**: 500 lux minimum at floor level provides adequate visibility for human operators. Even illumination reduces shadows that can confuse vision algorithms.

**Task Lighting**: Additional lighting over workstations and workbenches supports detailed work. Adjustable task lights allow workers to optimize lighting for their specific needs.

**Machine Vision Lighting**: Dedicated lighting for vision systems provides consistent illumination for cameras. Ring lights, bar lights, and dome lights each have advantages for different applications. Color temperature consistency is important for color-calibrated vision systems.

### Noise Control

Robotic systems generate noise from motors, fans, and mechanical systems. While this noise is typically below hearing-damage levels, it can be fatiguing during long sessions:

**Acoustic Treatment**: Absorptive materials on walls and ceilings reduce reverberation and overall noise levels. Foam panels and acoustic tiles are effective and relatively inexpensive.

**Enclosure**: Noisy equipment—high-powered simulation workstations, for example—can be enclosed in acoustic cabinets that reduce noise while maintaining airflow.

**Scheduling**: Coordinate loud activities to avoid overlapping with focused work periods. Running simulation jobs overnight when no one is present is better than running them during the day.

## Standard Operating Procedures

Beyond physical infrastructure, effective laboratories establish and enforce standard operating procedures (SOPs) that govern daily activities.

### Pre-Operation Checks

Before any robot testing, operators should verify:

1. **E-stop systems** are functional and accessible
2. **Perimeter** is clear of unauthorized personnel
3. **Emergency lighting** is functional
4. **Communication systems** (radios, intercoms) are working
5. **First aid kit** is accessible and stocked
6. **Environmental conditions** (temperature, lighting) are adequate

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class PreOpChecklist:
    """Pre-operation checklist for robot testing."""
    timestamp: datetime
    operator: str
    robot_id: str

    estop_functional: bool = False
    estop_accessible: bool = False
    perimeter_clear: bool = False
    emergency_lighting_ok: bool = False
    communication_ok: bool = False
    first_aid_accessible: bool = False
    temperature_ok: bool = False
    lighting_ok: bool = False

class ChecklistManager:
    """
    Manages safety checklists for laboratory operations.
    """

    def __init__(self):
        self.checklists: List[PreOpChecklist] = []
        self.current_checklist: Optional[PreOpChecklist] = None

    def start_checklist(self, operator: str, robot_id: str) -> PreOpChecklist:
        """Start a new pre-operation checklist."""
        self.current_checklist = PreOpChecklist(
            timestamp=datetime.now(),
            operator=operator,
            robot_id=robot_id
        )
        return self.current_checklist

    def complete_item(self, item_name: str, status: bool) -> bool:
        """Complete a checklist item."""
        if not self.current_checklist:
            return False

        if hasattr(self.current_checklist, item_name):
            setattr(self.current_checklist, item_name, status)
            return True
        return False

    def submit_checklist(self) -> Dict:
        """Submit completed checklist and get authorization."""
        if not self.current_checklist:
            return {'error': 'No active checklist'}

        all_items = [
            'estop_functional', 'estop_accessible', 'perimeter_clear',
            'emergency_lighting_ok', 'communication_ok', 'first_aid_accessible',
            'temperature_ok', 'lighting_ok'
        ]

        completed = sum(1 for item in all_items
                       if getattr(self.current_checklist, item_name, False))

        status = 'CLEARED' if completed == len(all_items) else 'NOT_CLEARED'

        result = {
            'status': status,
            'items_completed': f"{completed}/{len(all_items)}",
            'timestamp': self.current_checklist.timestamp.isoformat(),
            'operator': self.current_checklist.operator,
            'robot_id': self.current_checklist.robot_id
        }

        self.checklists.append(self.current_checklist)
        self.current_checklist = None

        return result

# Example checklist usage
manager = ChecklistManager()

print("Pre-Operation Checklist")
print("=" * 50)

# Start checklist
checklist = manager.start_checklist(operator="engineer_1", robot_id="humanoid_01")

# Complete items (in real usage, this would be interactive)
items_to_complete = [
    ('estop_functional', True),
    ('estop_accessible', True),
    ('perimeter_clear', True),
    ('emergency_lighting_ok', True),
    ('communication_ok', True),
    ('first_aid_accessible', True),
    ('temperature_ok', True),
    ('lighting_ok', True)
]

for item, status in items_to_complete:
    manager.complete_item(item, status)

# Submit and get authorization
result = manager.submit_checklist()
print(f"Authorization Status: {result['status']}")
print(f"Items Completed: {result['items_completed']}")
```

### Incident Response

When incidents occur—whether minor equipment damage or serious injuries—clear response protocols save time and prevent escalation:

**Minor Incidents**: Document the incident, assess damage, and conduct root cause analysis before resuming operations. Near-misses should be treated as seriously as actual incidents.

**Major Incidents**: Secure the area, provide first aid, and notify emergency services if needed. Do not attempt to move injured persons or significantly damaged equipment. Preserve the scene for investigation.

**After Action Review**: Every incident, regardless of severity, should trigger a review that identifies causes and preventive measures. These reviews should be blameless—the goal is learning, not punishment.

## Key Takeaways

A well-designed laboratory enables productive robotics research by providing the physical infrastructure, safety systems, and organizational structures that complex humanoid robot development requires. The investment in proper facilities pays dividends throughout a project in reduced debugging time, improved safety outcomes, and better research results.

The key principles to remember are:

- **Space for failure**: Robots will fall; plan for it with generous clear zones and protective surfaces.
- **Defense in depth**: Layer multiple safety systems so that single-point failures don't cause accidents.
- **Infrastructure first**: Invest in power, networking, and data systems before purchasing robots.
- **Procedures matter**: Physical infrastructure without supporting procedures is insufficient.

With a safe, well-equipped laboratory established, we can now explore the broader landscape of humanoid robotics—understanding where this technology came from and where it's headed.
