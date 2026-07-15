# GrSim-Football-Simulator
Three-level ROS 2 pipelined autonomous grSim small size league (SSL) holonomic striker agent Contains robocaster-based proportional motion that combines Khatib APFs with tangential force blends for obstacle avoidance, and a tailored protobuf fallback between the two encoders.

# Autonomous SSL Soccer Agent (ROS 2)

An autonomous perception-planning-control pipeline for a RoboCup Small Size League (SSL) striker agent. Built on **ROS 2** and integrated with the **grSim** physics simulator, this system uses a three-tier architecture to process high-frequency SSL-Vision data and execute high-speed, obstacle-resilient navigation.

---

## 🚀 Features

* **Three-Tier Architecture:** Seamless translation from raw SSL-Vision UDP packets to ROS 2 messages (`ssl_league_msgs`), down to low-level motor actuation.
* **Holonomic Kinematics:** 2D coordinate-frame rotation mapping global field positions into local body-frame velocities ($v_x$, $v_y$, $\omega$).
* **Reactive Obstacle Avoidance (APF):** Implements Khatib's Artificial Potential Fields with a **70% tangential / 30% radial** force blend to steer smoothly around opponents without halting.
* **Local Minima Recovery:** A time-based stuck-detector that overrides deadlocks with a randomized lateral escape impulse.
* **Robust Custom Protobuf Encoder:** Includes a hand-crafted Python fallback serializer (varint/fixed32) to generate valid `RobotControl` packets without relying on compiled system dependencies.

---

## 📐 Mathematical Formulation

### 1. Coordinate Transformation
Global position errors are rotated into the robot's local frame using its yaw ($\theta$) extracted from the vision quaternion:

$$
\begin{bmatrix}
v_{x\_local} \\
v_{y\_local}
\end{bmatrix} = 
\begin{bmatrix}
\cos(\theta) & \sin(\theta) \\
-\sin(\theta) & \cos(\theta)
\end{bmatrix}
\begin{bmatrix}
err_x \\
err_y
\end{bmatrix}
$$

### 2. Control Law
A decoupled Proportional (P) controller drives translational and rotational channels independently:
* **Translational Gain ($k_{p\_trans}$):** $2.0$ (saturated at $1.3 \sim 1.5 \text{ m/s}$)
* **Rotational Gain ($k_{p\_rot}$):** $4.0$ (clamped at $\pm 3.5 \text{ rad/s}$)
* **Deadbands:** $0.05 \text{ m}$ (position) and $0.05 \text{ rad}$ (heading) to suppress limit-cycle oscillations.

### 3. Obstacle Avoidance (APF)
Attractive forces scale linearly near the goal ($d \le 0.60\text{ m}$) and bound to a constant magnitude in the far field ($d > 0.60\text{ m}$). 

Repulsive force vectors ($F_{rep}$) are calculated using a Khatib formulation modified with a lateral bias to safely thread narrow gaps:

$$F_{rep} = \text{rep\_mag} \cdot \left[ (1 - w)\vec{u}_{radial} + w(\vec{u}_{tangential}) \right]$$

where $w = 0.70$ (tangential weight) and the influence radius $\rho_0 = 0.60\text{ m}$.

---

## 🏗️ System Architecture


### Node Explanations:
1.  **`teleop_node` (Phase 1):** Validates the communication stack. Non-blocking keyboard teleoperation running on a background thread at 50 Hz.
2.  **`tracker_node` (Phase 2):** Autonomous tracking. Drives the robot to a target point computed $0.30\text{ m}$ offset behind the ball aligned with the opponent's goal center.
3.  **`navigator_node` (Phase 3):** Full reactive navigation. Integrates the APF obstacle avoidance and a 30-cycle positional snapshot stuck-detector to bypass complex opponent configurations.

---

## 🛠️ Setup & Installation

+-----------------------+
|    grSim Simulator    | <---------+
+-----------------------+           |
            |                       |
    UDP (SSL-Vision)                |
            v                       |
+----------------------------+      |
|      ssl_ros_bridge        |      | UDP Protobuf
+----------------------------+      | (RobotControl)
            |                       |
  ROS 2 (/ssl_vision_bridge)        |
            v                       |
+----------------------------+      |
|        Control Node        | -----+
| (teleop / tracker / nav)   |
+----------------------------+

### Prerequisites
* Ubuntu 22.04 LTS (or equivalent)
* ROS 2 (Humble/Iron/Jazzy)
* [grSim](https://github.com/RoboCup-SSL/grSim) Simulator
* `ssl_league_msgs` ROS 2 package
