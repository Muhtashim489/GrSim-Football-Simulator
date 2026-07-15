#!/usr/bin/env python3
"""
Phase 4: Autonomous Strike — State Machine Control with Integrated Multi-Team APF Avoidance
FIXED: Corrected team array mapping to ensure Blue team correctly avoids Yellow obstacles.
"""

import math
import random
import socket
import struct
import rclpy
from rclpy.node import Node
from ssl_league_msgs.msg import VisionWrapper

GRSIM_HOST        = '127.0.0.1'
GRSIM_BLUE_PORT   = 10301
GRSIM_YELLOW_PORT = 10302

try:
    from ssl_league_protobufs import ssl_simulation_robot_control_pb2 as robot_ctrl_pb
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False


def quat_to_yaw(qx, qy, qz, qw):
    """Extract yaw (rotation around Z) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a):
    """Normalize angles to [-pi, pi]."""
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def build_udp_packet(robot_id, vx, vy, vw):
    """Encodes robot velocity command using protobuf if available, else falls back to manual packing."""
    if PROTOBUF_AVAILABLE:
        ctrl = robot_ctrl_pb.RobotControl()
        cmd  = ctrl.robot_commands.add()
        cmd.id = robot_id
        cmd.move_command.local_velocity.forward = vx
        cmd.move_command.local_velocity.left    = vy
        cmd.move_command.local_velocity.angular = vw
        cmd.kick_speed = cmd.kick_angle = cmd.dribbler_speed = 0.0
        return ctrl.SerializeToString()

    def varint(v):
        out = b''
        while True:
            bits = v & 0x7F; v >>= 7
            out += bytes([bits | (0x80 if v else 0)])
            if not v: break
        return out
    def pfloat(tag, val):
        return varint(tag << 3 | 5) + struct.pack('<f', val)
    local_vel  = pfloat(1, vx) + pfloat(2, vy) + pfloat(3, vw)
    move_inner = varint(2<<3|2) + varint(len(local_vel))   + local_vel
    robot_cmd  = varint(1<<3|0) + varint(robot_id) + varint(2<<3|2) + varint(len(move_inner)) + move_inner
    return varint(1<<3|2) + varint(len(robot_cmd)) + robot_cmd


class SSLAutonomousStrikeNode(Node):

    def __init__(self):
        super().__init__('ssl_autonomous_strike_node')

        # ROS 2 Parameters
        self.declare_parameter('robot_id', 2)
        self.declare_parameter('is_team_yellow', False) # Default to False if you are launching as Blue team
        self.declare_parameter('offset_distance', 0.15)    
        self.declare_parameter('grsim_host', GRSIM_HOST)

        self.robot_id        = self.get_parameter('robot_id').value
        self.is_team_yellow  = self.get_parameter('is_team_yellow').value
        self.offset_distance = self.get_parameter('offset_distance').value
        self.grsim_host      = self.get_parameter('grsim_host').value

        # Dynamic Team Target Assignment
        if self.is_team_yellow:
            self.opponent_goal_x = -4.5
            self.get_logger().info("🟨 YELLOW TEAM Active: Attacking the Blue Net at X = -4.5")
        else:
            self.opponent_goal_x = 4.5
            self.get_logger().info("🟦 BLUE TEAM Active: Attacking the Yellow Net at X = +4.5")
        
        self.grsim_port = GRSIM_YELLOW_PORT if self.is_team_yellow else GRSIM_BLUE_PORT

        # State Machine Configurations
        self.STATE_APPROACH  = "APPROACH_BALL"
        self.STATE_STRIKE    = "STRIKE"
        self.current_state   = self.STATE_APPROACH

        # APF Obstacle Tuning Constants
        self.k_att = 2.8       
        self.k_rep = 4.5        # Slightly higher repulsion to handle high-speed launch scenarios
        self.rho_0 = 0.65       # Increased activation zone to give the robot more reaction time
        self.tangent_weight = 0.70 

        # Speed Thresholds and Deadbands
        self.max_linear_speed  = 1.5   
        self.max_angular_speed = 4.5
        self.kp_rot            = 5.5
        self.pos_deadband      = 0.05
        self.rot_deadband      = 0.05
        
        # State Transition Thresholds
        self.strike_zone_dist  = 0.25  
        self.alignment_thresh  = 0.40  

        # Phase 3 Local Minimum Escape States
        self.escape_speed          = 0.6
        self.escape_duration       = 15
        self.stuck_dist_threshold  = 0.015
        self.stuck_check_interval  = 20
        
        self._escape_vy    = 0.0
        self._escape_ticks = 0
        self._snap_x       = 0.0
        self._snap_y       = 0.0
        self._loop_counter = 0

        # Dynamic World States
        self.robot_x = self.robot_y = self.robot_theta = None
        self.ball_x = self.ball_y = None
        self.obstacles_list = []  

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.vision_sub = self.create_subscription(
            VisionWrapper, '/ssl_vision_bridge/vision_messages', self.vision_callback, 10
        )

        self.get_logger().info(f"Phase 4 Striker Node Operational. Target Net X: {self.opponent_goal_x}")

    def vision_callback(self, msg: VisionWrapper):
        if not msg.detection:
            return

        best_ball_conf = -1.0
        best_striker_conf = -1.0
        obstacles_candidate = []

        for frame in msg.detection:
            # 1. Parse Ball Data
            for ball in frame.balls:
                if ball.confidence > best_ball_conf:
                    best_ball_conf = ball.confidence
                    self.ball_x = ball.pos.x
                    self.ball_y = ball.pos.y

            # --- CORRECTION STAGE ---
            # Dynamically determine array tracking roles based on active team identity
            if self.is_team_yellow:
                my_team_robots = frame.robots_yellow
                opponent_team_robots = frame.robots_blue
            else:
                my_team_robots = frame.robots_blue
                opponent_team_robots = frame.robots_yellow

            # 2. Parse Teammates (Skip self, handle others)
            for r in my_team_robots:
                if r.robot_id == self.robot_id and r.confidence > best_striker_conf:
                    best_striker_conf = r.confidence
                    self.robot_x     = r.pose.position.x
                    self.robot_y     = r.pose.position.y
                    q                = r.pose.orientation
                    self.robot_theta = quat_to_yaw(q.x, q.y, q.z, q.w)
                elif r.robot_id != self.robot_id and r.confidence > 0.3:
                    obstacles_candidate.append({'x': r.pose.position.x, 'y': r.pose.position.y})

            # 3. Parse Opponents
            for r in opponent_team_robots:
                if r.confidence > 0.3:
                    obstacles_candidate.append({'x': r.pose.position.x, 'y': r.pose.position.y})

        # Drop duplicate vision frames to prevent potential field inflation
        self.obstacles_list = []
        seen_positions = set()
        for obs in obstacles_candidate:
            pos_key = (round(obs['x'], 2), round(obs['y'], 2))
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                self.obstacles_list.append(obs)

        if self.ball_x is None or self.robot_x is None or self.robot_theta is None:
            return

        self._autonomous_control_loop()

    def _autonomous_control_loop(self):
        # 1. Coordinate Vector Calculations
        dx_r2b = self.ball_x - self.robot_x
        dy_r2b = self.ball_y - self.robot_y
        dist_to_ball = math.hypot(dx_r2b, dy_r2b)

        # Vector pointing from Target Goal Net down to the Ball
        dx_g2b = self.ball_x - self.opponent_goal_x
        dy_g2b = self.ball_y - 0.0 
        dist_g2b = math.hypot(dx_g2b, dy_g2b)

        ux_g2b = dx_g2b / max(0.01, dist_g2b)
        uy_g2b = dy_g2b / max(0.01, dist_g2b)

        # Setup coordinate target positioned cleanly behind the ball along the goal line vector
        target_setup_x = self.ball_x + self.offset_distance * ux_g2b
        target_setup_y = self.ball_y + self.offset_distance * uy_g2b

        dx_to_setup = target_setup_x - self.robot_x
        dy_to_setup = target_setup_y - self.robot_y
        dist_to_setup = math.hypot(dx_to_setup, dy_to_setup)

        # Calculate Headings
        heading_to_ball = math.atan2(self.ball_y - self.robot_y, self.ball_x - self.robot_x)
        heading_to_goal = math.atan2(0.0 - self.robot_y, self.opponent_goal_x - self.robot_x)
        alignment_error = abs(normalize_angle(heading_to_goal - self.robot_theta))

        # 2. State Machine Arbitration
        if self.current_state == self.STATE_APPROACH:
            standard_strike = (dist_to_ball < self.strike_zone_dist and alignment_error < self.alignment_thresh)
            boundary_override = (dist_to_ball < 0.13)

            if standard_strike or boundary_override:
                self.current_state = self.STATE_STRIKE
                self.get_logger().info("TARGET LOCKED: Transitioning to STRIKE Behavior.")
                self._escape_ticks = 0
        
        elif self.current_state == self.STATE_STRIKE:
            if dist_to_ball > (self.strike_zone_dist * 2.8):
                self.current_state = self.STATE_APPROACH
                self.get_logger().info("BALL DEFLECTED: Reverting to APPROACH tracking.")

        # 3. Formulate Attraction Vectors
        if self.current_state == self.STATE_APPROACH:
            f_att_x = self.k_att * dx_to_setup
            f_att_y = self.k_att * dy_to_setup

            # Approach Speed Dampening near touch point
            if dist_to_setup < 0.20:
                dampen_factor = max(0.35, dist_to_setup / 0.20)
                f_att_x *= dampen_factor
                f_att_y *= dampen_factor
            
            desired_heading = heading_to_ball  
            current_target_dist = dist_to_setup
        else:
            dx_att = self.opponent_goal_x - self.robot_x
            dy_att = 0.0 - self.robot_y
            dist_to_goal = max(0.01, math.hypot(dx_att, dy_att))
            
            f_att_x = (self.k_att * 3.5) * (dx_att / dist_to_goal)
            f_att_y = (self.k_att * 3.5) * (dy_att / dist_to_goal)
            
            desired_heading = heading_to_ball  
            current_target_dist = dist_to_ball

        # 4. Corrected APF Obstacle Avoidance Forces
        f_rep_x = 0.0
        f_rep_y = 0.0

        for obs in self.obstacles_list:
            dx_r2o = self.robot_x - obs['x']
            dy_r2o = self.robot_y - obs['y']
            dist_to_obs = math.hypot(dx_r2o, dy_r2o)

            if dist_to_obs < self.rho_0 and dist_to_obs > 0.01:
                # Radial Repulsion Force
                rep_magnitude = self.k_rep * (1.0 / dist_to_obs - 1.0 / self.rho_0) / (dist_to_obs ** 2)
                ux_r2o = dx_r2o / dist_to_obs
                uy_r2o = dy_r2o / dist_to_obs

                f_rep_x += rep_magnitude * ux_r2o
                f_rep_y += rep_magnitude * uy_r2o

                # Tangential component to push around corners smoothly
                f_rep_x += self.tangent_weight * rep_magnitude * (-uy_r2o)
                f_rep_y += self.tangent_weight * rep_magnitude * (ux_r2o)

        # Mix total vectors together
        f_total_x = f_att_x + f_rep_x
        f_total_y = f_att_y + f_rep_y

        # 5. Transform Global World Vectors into Robot Local Frame
        cos_t = math.cos(self.robot_theta)
        sin_t = math.sin(self.robot_theta)
        vx_cmd = f_total_x * cos_t + f_total_y * sin_t
        vy_cmd = -f_total_x * sin_t + f_total_y * cos_t

        err_theta = normalize_angle(desired_heading - self.robot_theta)
        vw_cmd = self.kp_rot * err_theta

        # 6. Apply Phase 3 Local Minimum Escape Logic
        if self._escape_ticks > 0:
            vy_cmd = self._escape_vy
            vx_cmd = 0.1  
            self._escape_ticks -= 1
        else:
            if self.current_state == self.STATE_APPROACH:
                self._loop_counter += 1
                if self._loop_counter >= self.stuck_check_interval:
                    self._loop_counter = 0
                    if self.robot_x is not None:
                        moved = math.hypot(self.robot_x - self._snap_x, self.robot_y - self._snap_y)
                        if moved < self.stuck_dist_threshold and current_target_dist > self.pos_deadband * 3:
                            side = random.choice([-1.0, 1.0])
                            self._escape_vy    = side * self.escape_speed
                            self._escape_ticks = self.escape_duration
                    self._snap_x = self.robot_x
                    self._snap_y = self.robot_y

        # 7. Speed Saturation Limits & Deadbands
        speed = math.hypot(vx_cmd, vy_cmd)
        if speed > self.max_linear_speed:
            scale = self.max_linear_speed / speed
            vx_cmd *= scale
            vy_cmd *= scale

        vw_cmd = max(-self.max_angular_speed, min(self.max_angular_speed, vw_cmd))

        # Deadband Dampeners
        if current_target_dist < self.pos_deadband and self._escape_ticks == 0 and self.current_state == self.STATE_APPROACH:
            vx_cmd = vy_cmd = 0.0
        if abs(err_theta) < self.rot_deadband:
            vw_cmd = 0.0

        # 8. Fire Commands Over UDP Channel to grSim
        try:
            pkt = build_udp_packet(self.robot_id, vx_cmd, vy_cmd, vw_cmd)
            self.sock.sendto(pkt, (self.grsim_host, self.grsim_port))
        except Exception as e:
            self.get_logger().error(f"UDP frame transmission failure: {e}")

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SSLAutonomousStrikeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()