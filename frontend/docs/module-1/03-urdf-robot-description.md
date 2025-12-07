---
sidebar_position: 3
---

# URDF Robot Description

## Introduction
Unified Robot Description Format (URDF) is the standard for describing robots in ROS. This module covers the creation and use of URDF files to define robot geometry, kinematics, and dynamics.

## Learning Objectives
- Understand URDF structure and elements
- Create robot models with links and joints
- Define visual and collision properties
- Incorporate inertial properties for simulation

## URDF Fundamentals
- XML-based format structure
- Links: rigid bodies of the robot
- Joints: connections between links
- Materials and colors

## Link Elements
- Visual elements (geometry, material, origin)
- Collision elements
- Inertial properties (mass, inertia matrix)
- Multiple visual/collision elements per link

## Joint Types
- Fixed joints
- Revolute joints (rotational)
- Prismatic joints (linear)
- Continuous joints
- Floating and planar joints

## Advanced Features
- Transmission elements
- Gazebo-specific elements
- Robot semantic description (SRDF)
- Xacro macro language

## Validation and Tools
- URDF validation tools
- Visualization with RViz
- Kinematic analysis
- Export/import considerations

## Best Practices
- Proper scaling and units
- Appropriate collision geometry
- Realistic inertial properties
- Modular design approaches

## Summary
URDF provides a comprehensive framework for describing robot geometry and kinematics in ROS.