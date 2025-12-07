---
sidebar_position: 1
---

# ROS 2 Architecture

## Introduction
Robot Operating System 2 (ROS 2) serves as the nervous system for robotic applications, providing a flexible framework for developing robot software. Unlike its predecessor ROS 1, which was built on a peer-to-peer graph architecture using TCPROS and XMLRPC, ROS 2 is built on DDS (Data Distribution Service) and provides enhanced features including improved real-time support, security, and multi-robot systems.

ROS 2 addresses many of the limitations of ROS 1, particularly in production environments. It provides better support for multi-robot systems, improved security, real-time capabilities, and cross-platform compatibility. The architecture is designed to be more robust for deployment in real-world applications, from manufacturing to service robotics.

## Learning Objectives
- Understand the fundamental architecture of ROS 2 and how it differs from ROS 1
- Learn about nodes, topics, services, and actions and their roles in ROS 2 systems
- Explore the middleware layer and communication mechanisms that enable ROS 2's distributed architecture
- Compare key differences between ROS 1 and ROS 2 and understand when to use each
- Grasp the concept of Quality of Service (QoS) policies and their importance in robotic systems

## ROS 2 Architecture Overview

### Client Libraries (rclcpp, rclpy)
ROS 2's client libraries provide the interface between user code and the underlying middleware. The most commonly used libraries are:

- **rclcpp**: The C++ client library that provides C++ interfaces for ROS 2 concepts. It's built on top of the ROS Client Library (rcl) and provides efficient, type-safe interfaces for nodes, publishers, subscribers, and other ROS 2 entities.

- **rclpy**: The Python client library that provides Python interfaces for ROS 2 concepts. It uses Python's asyncio for asynchronous operations and provides a more Pythonic interface compared to rclcpp.

- **Other languages**: ROS 2 also supports Rust, Java, and other languages through additional client libraries, making it more accessible to developers from different backgrounds.

The client libraries handle serialization, deserialization, and communication with the middleware layer, abstracting away the complexity of DDS implementation details.

### ROS Middleware (RMW) Layer
The ROS Middleware (RMW) layer is a critical component that abstracts the underlying DDS implementation. This abstraction allows ROS 2 to work with different DDS implementations without changing the user-facing API. The RMW layer:

- Provides a common interface for different DDS implementations
- Handles the translation between ROS 2 concepts and DDS concepts
- Manages the lifecycle of DDS entities
- Implements Quality of Service (QoS) policy mapping
- Handles type support and message serialization

Multiple DDS implementations can be used with ROS 2, including Fast DDS, Cyclone DDS, RTI Connext DDS, and OpenSplice DDS.

### DDS Implementations
Data Distribution Service (DDS) is a middleware protocol and API standard for distributed, real-time, publish-subscribe communication. In ROS 2, DDS provides:

- **Discovery**: Automatic discovery of participants in the network
- **Communication**: Publish-subscribe, request-reply communication patterns
- **Quality of Service**: Configurable policies for reliability, durability, and other aspects
- **Real-time capabilities**: Support for real-time systems with deterministic behavior
- **Fault tolerance**: Built-in mechanisms for handling network failures and participant failures

The choice of DDS implementation can affect performance, real-time capabilities, and feature availability in ROS 2 systems.

### Node Composition and Lifecycle
ROS 2 introduces the concept of node composition, which allows multiple nodes to be run in the same process. This is achieved through:

- **Components**: Reusable pieces of functionality that can be composed into nodes
- **Lifecycle nodes**: Nodes that have explicit state management for initialization, configuration, activation, and shutdown
- **Node parameters**: Dynamic configuration parameters that can be changed at runtime
- **Namespaces**: Hierarchical organization of nodes, topics, and services

This architecture enables more efficient resource utilization and better integration of different components within a robotic system.

## Core Concepts

### Nodes: Processes that perform computation
Nodes in ROS 2 are processes that perform computation and are the fundamental building blocks of a ROS system. Key characteristics include:

- **Encapsulation**: Each node encapsulates specific functionality within a robotic system
- **Communication**: Nodes communicate with each other through topics, services, and actions
- **Namespacing**: Nodes can be organized using namespaces for better organization
- **Parameters**: Nodes can have configurable parameters that can be set at runtime
- **Lifecycle**: Nodes can implement lifecycle interfaces for explicit state management

Nodes in ROS 2 are more robust than in ROS 1, with better error handling and process isolation.

### Topics: Publish/Subscribe Communication
Topics in ROS 2 implement a publish/subscribe communication pattern where:

- **Publishers** send messages to topics without knowing which subscribers exist
- **Subscribers** receive messages from topics without knowing which publishers exist
- **Decoupling**: Publishers and subscribers are decoupled in time and space
- **Data flow**: Unidirectional data flow from publishers to subscribers
- **Message types**: Each topic has a specific message type that defines its structure

Topics are ideal for streaming data like sensor readings, robot states, or other continuous information flows.

### Services: Request/Response Communication
Services in ROS 2 implement a request/response communication pattern where:

- **Clients** send a request and wait for a response
- **Servers** receive requests and send back responses
- **Synchronous**: Communication is typically synchronous with guaranteed delivery
- **Request/Response types**: Each service has specific types for requests and responses
- **Blocking**: Client calls are typically blocking until a response is received

Services are suitable for operations that require a specific response, like triggering actions or querying system state.

### Actions: Goal-Based Communication with Feedback
Actions in ROS 2 provide a more sophisticated communication pattern that includes:

- **Goals**: Requests for long-running operations with specific objectives
- **Feedback**: Continuous updates on the progress of goal execution
- **Results**: Final outcomes of goal execution
- **Preemption**: Ability to cancel or replace goals in progress
- **Status tracking**: Detailed status information throughout the process

Actions are ideal for complex tasks like navigation, manipulation, or any operation that takes time and requires monitoring.

## Communication Patterns

### Publisher/Subscriber Pattern
The publisher/subscriber pattern in ROS 2 enables:

- **Asynchronous communication**: Publishers and subscribers operate independently
- **Multiple participants**: Multiple publishers and subscribers can interact with the same topic
- **Real-time capabilities**: Low-latency communication suitable for real-time systems
- **Reliability options**: Different reliability policies for various use cases
- **Transport flexibility**: Support for various transport mechanisms (UDP, TCP, shared memory)

This pattern is fundamental to ROS 2's distributed architecture and enables the creation of complex robotic systems from simple components.

### Client/Server Pattern
The client/server pattern provides:

- **Synchronous communication**: Request/response interactions with guaranteed delivery
- **Service discovery**: Automatic discovery of available services
- **Type safety**: Compile-time checking of request/response types
- **Timeout handling**: Configurable timeouts for service calls
- **Multi-server support**: Multiple servers can provide the same service

This pattern is essential for operations requiring guaranteed responses and state changes.

### Action-Based Communication
Action-based communication offers:

- **Long-running operations**: Support for operations that take significant time
- **Progress monitoring**: Continuous feedback on operation progress
- **Goal management**: Ability to send, cancel, and monitor multiple goals
- **Result delivery**: Reliable delivery of operation results
- **State management**: Built-in state management for complex operations

Actions are particularly useful for robotics tasks like navigation, where you need to monitor progress and potentially cancel operations.

### Parameter Server Functionality
ROS 2 provides enhanced parameter functionality:

- **Node parameters**: Each node can declare and manage its own parameters
- **Dynamic reconfiguration**: Parameters can be changed at runtime
- **Parameter services**: Standardized services for parameter management
- **Parameter callbacks**: Ability to react to parameter changes
- **Parameter validation**: Validation of parameter values before acceptance

This enables more flexible and dynamic system configuration.

## Quality of Service (QoS)

### Reliability and Durability Policies
QoS policies in ROS 2 provide:

- **Reliability**: Controls whether messages are guaranteed to be delivered
  - `RELIABLE`: All messages are guaranteed to be delivered
  - `BEST_EFFORT`: Messages are sent without guarantee of delivery
- **Durability**: Controls how messages are handled for late-joining subscribers
  - `TRANSIENT_LOCAL`: Messages are stored for late joiners from the same node
  - `VOLATILE`: Messages are not stored for late joiners

### Deadline and Lifespan Settings
- **Deadline**: Maximum time between data samples
- **Lifespan**: Maximum lifetime of data samples
- **Liveliness**: Mechanism to ensure participants are active
- **History**: Policy for storing data samples
  - `KEEP_LAST`: Store the most recent samples
  - `KEEP_ALL`: Store all samples (limited by resource limits)

### History and Depth Configurations
QoS history policies determine how many samples are stored:

- **Depth**: Number of samples to store for KEEP_LAST policy
- **Resource limits**: Memory and time constraints for data storage
- **Compatibility**: Ensuring QoS compatibility between publishers and subscribers

### Matching Policies
- **Publisher/Subscriber matching**: Rules for connecting publishers and subscribers
- **Compatibility checking**: Validation of QoS policy compatibility
- **Configuration flexibility**: Ability to adjust policies for different use cases

## Security Features

### Authentication and Authorization
ROS 2 security framework includes:

- **Identity management**: Secure identification of nodes and participants
- **Certificate-based authentication**: X.509 certificates for participant authentication
- **Access control**: Authorization policies for resource access
- **Secure discovery**: Protection of the discovery process

### Encryption of Data
- **Transport encryption**: End-to-end encryption of data in transit
- **Message encryption**: Optional encryption of individual messages
- **Key management**: Secure handling of encryption keys
- **Algorithm flexibility**: Support for various encryption algorithms

### Secure Communication Channels
- **Secure DDS**: Implementation of DDS security specifications
- **Secure endpoints**: Protection of communication endpoints
- **Secure multicast**: Secure handling of multicast communication
- **Secure logging**: Protection of system logs and diagnostics

### Identity Management
- **Participant authentication**: Verification of participant identity
- **Node authentication**: Verification of individual node identity
- **Certificate management**: Handling of digital certificates
- **Trust relationships**: Management of trust between participants

## Summary
ROS 2 provides a robust and flexible architecture for building complex robotic systems. Its foundation on DDS enables improved real-time capabilities, security, and multi-robot system support compared to ROS 1. The architecture's modular design with client libraries, middleware abstraction, and comprehensive communication patterns makes it suitable for both research and production environments.

The introduction of Quality of Service policies, lifecycle nodes, and enhanced security features addresses many of the limitations of ROS 1, making ROS 2 a more mature platform for real-world robotic applications. Understanding these architectural concepts is crucial for developing efficient, reliable, and scalable robotic systems using ROS 2.