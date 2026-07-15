#!/usr/bin/env python3
"""
Phase 4: Autonomous Strike Node (Realistic Striker & Continuous Dribble-to-Goal)
Behavior Flow:
  1. GET_BEHIND_BALL: Navigates behind the ball using APF obstacle avoidance.
  2. DRIBBLE_TO_GOAL: Activates dribbler, grabs the ball, and drives it towards 
     the opponent's goal while actively dodging obstacles dynamically.
  3. STRIKE: Once near the goal, executes a maximum hardware kick.
"""

import sys
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


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(a):
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return math.atan2(math.sin(a), math.cos(a))

def build_udp_packet(robot_id, vx, vy, vw, kick_speed=0.0, dribbler_speed=0.0):
    if PROTOBUF_AVAILABLE:
        ctrl = robot_ctrl_pb.RobotControl()
        cmd  = ctrl.robot_commands.add()
        cmd.id = robot_id
        cmd.move_command.local_velocity.forward = vx
        cmd.move_command.local_velocity.left    = vy
        cmd.move_command.local_velocity.angular = vw
        cmd.kick_speed = kick_speed
        cmd.kick_angle = 0.0
        cmd.dribbler_speed = dribbler_speed
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
    move_inner = varint(2<<3|2) + varint(len(local_vel)) + local_vel
    
    robot_cmd  = varint(1<<3|0) + varint(robot_id) + varint(2<<3|2) + varint(len(move_inner)) + move_inner
    if kick_speed > 0.0:
        robot_cmd += pfloat(3, kick_speed)
    if dribbler_speed > 0.0:
        robot_cmd += pfloat(5, dribbler_speed)
        
    return varint(1<<3|2) + varint(len(robot_cmd)) + robot_cmd


class FSMState:
    GET_BEHIND_BALL = 0
    DRIBBLE_TO_GOAL = 1
    STRIKE = 2


class SSLAutonomousNode(Node):

    def __init__(self):
        super().__init__('ssl_autonomous_node')

        # Declare parameters
        self.declare_parameter('robot_id',       0)
        self.declare_parameter('is_team_yellow', True)  # Yellow = True, Blue = False
        self.declare_parameter('offset_distance', 0.22) # Precise distance behind ball (m)
        self.declare_parameter('vision_topic',   '/ssl_vision_bridge/vision_messages')
        self.declare_parameter('grsim_host',     GRSIM_HOST)

        self.robot_id        = self.get_parameter('robot_id').value
        self.is_team_yellow  = self.get_parameter('is_team_yellow').value
        self.offset_distance = self.get_parameter('offset_distance').value
        vision_topic_name    = self.get_parameter('vision_topic').value
        self.grsim_host      = self.get_parameter('grsim_host').value
        self.grsim_port      = GRSIM_YELLOW_PORT if self.is_team_yellow else GRSIM_BLUE_PORT

        # TEAM SIDE DETECTION: Yellow attacks negative X (-4.5), Blue attacks positive X (+4.5)
        self.opponent_goal_x = -4.5 if self.is_team_yellow else 4.5

        # ── APF Evasion & Driving Parameters ──────────────────────────────────
        self.k_att           = 4.0   # Fast attraction to target
        self.k_rep           = 5.5   # Aggressive repulsion to bounce off obstacles
        self.rho_0           = 0.70  # Wide obstacle safety bubble (m)
        self.d_switch        = 0.4
        self.tangent_weight  = 0.85  # Strong fluid sideways movement around defenders

        self.max_linear_speed  = 2.0
        self.max_angular_speed = 5.0
        self.kp_rot            = 6.0

        # ── Stuck Detector ────────────────────────────────────────────────────
        self.stuck_check_interval  = 30
        self.stuck_dist_threshold  = 0.04
        self.escape_duration       = 15
        self.escape_speed          = 0.9
        self._callback_count  = 0
        self._snap_x          = None
        self._snap_y          = None
        self._escape_ticks    = 0
        self._escape_vy       = 0.0

        # ── State Machine Initialization ──────────────────────────────────────
        self.state = FSMState.GET_BEHIND_BALL
        self.state_labels = {
            FSMState.GET_BEHIND_BALL: "GET_BEHIND_BALL",
            FSMState.DRIBBLE_TO_GOAL: "DRIBBLE_TO_GOAL",
            FSMState.STRIKE: "STRIKE"
        }
        self.strike_tick_timer = 0

        # Vision State Variables
        self.robot_x = self.robot_y = self.robot_theta = None
        self.ball_x  = self.ball_y  = None
        self.opponents_list = []

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.vision_sub = self.create_subscription(
            VisionWrapper,
            vision_topic_name,
            self.vision_callback, 10
        )

        self.get_logger().info(
            f"Realistic Striker Loaded Successfully.\n"
            f"  Team: {'YELLOW (Attacking -4.5m)' if self.is_team_yellow else 'BLUE (Attacking +4.5m)'}\n"
            f"  Dribble & APF Obstacle Avoidance Matrix: ACTIVE."
        )

    def vision_callback(self, msg: VisionWrapper):
        if not msg.detection:
            return

        best_ball = best_striker = -1.0
        opponents_candidate = []

        for frame in msg.detection:
            # Ball tracking
            for ball in frame.balls:
                if ball.confidence > best_ball:
                    best_ball   = ball.confidence
                    self.ball_x = ball.pos.x
                    self.ball_y = ball.pos.y

            # Striker tracking
            strikers = frame.robots_yellow if self.is_team_yellow else frame.robots_blue
            for r in strikers:
                if r.robot_id == self.robot_id and r.confidence > best_striker:
                    best_striker     = r.confidence
                    self.robot_x     = r.pose.position.x
                    self.robot_y     = r.pose.position.y
                    q                = r.pose.orientation
                    self.robot_theta = quat_to_yaw(q.x, q.y, q.z, q.w)

            # Obstacle detection: ALL robots on the field except self
            # This handles both same-team and opposite-team defenders
            # BLUE robots
            for r in frame.robots_blue:
                is_self = (not self.is_team_yellow and r.robot_id == self.robot_id)

                if not is_self and r.confidence > 0.3:
                   opponents_candidate.append({
                   'team': 'blue',
                   'id': r.robot_id,
                   'x': r.pose.position.x,
                   'y': r.pose.position.y
                })

            # YELLOW robots
            for r in frame.robots_yellow:
                is_self = (self.is_team_yellow and r.robot_id == self.robot_id)

                if not is_self and r.confidence > 0.3:
                   opponents_candidate.append({
                   'team': 'yellow',
                   'id': r.robot_id,
                   'x': r.pose.position.x,
                   'y': r.pose.position.y
                })

        # Deduplicate by position proximity (same robot seen by multiple cameras)
        seen_ids = set()
        self.opponents_list = []
        for opp in opponents_candidate:
            key = f"{opp['id']}"
            if key not in seen_ids:
                seen_ids.add(key)
                self.opponents_list.append(opp)

        if self.ball_x is not None and self.robot_x is not None and self.robot_theta is not None:
            self._callback_count += 1
            self.process_state_machine()

    def process_state_machine(self):
        goal_x = self.opponent_goal_x
        goal_y = 0.0

        # Vector pointing from opponent goal to ball
        dx_g2b = self.ball_x - goal_x
        dy_g2b = self.ball_y - goal_y
        dist_g2b = max(math.hypot(dx_g2b, dy_g2b), 0.001)
        ux = dx_g2b / dist_g2b
        uy = dy_g2b / dist_g2b

        # Ideal alignment setup position behind ball
        target_x = self.ball_x + self.offset_distance * ux
        target_y = self.ball_y + self.offset_distance * uy

        dist_to_offset = math.hypot(target_x - self.robot_x, target_y - self.robot_y)
        dist_to_ball   = math.hypot(self.ball_x - self.robot_x, self.ball_y - self.robot_y)

        # Dynamic heading angle definitions
        angle_to_ball  = math.atan2(self.ball_y - self.robot_y, self.ball_x - self.robot_x)
        err_theta_ball = normalize_angle(angle_to_ball - self.robot_theta)

        angle_to_goal  = math.atan2(goal_y - self.robot_y, goal_x - self.robot_x)
        err_theta_goal = normalize_angle(angle_to_goal - self.robot_theta)

        global_fx, global_fy = 0.0, 0.0
        target_heading = err_theta_ball
        kick_speed = 0.0
        dribbler_speed = 0.0

        # ══════════════════════════════════════════════════════════════════════
        # State 0: GET_BEHIND_BALL (Go into position with APF Avoidance)
        # ══════════════════════════════════════════════════════════════════════
        if self.state == FSMState.GET_BEHIND_BALL:
            dx_att = target_x - self.robot_x
            dy_att = target_y - self.robot_y
            
            if dist_to_offset > self.d_switch:
                f_att_x = self.k_att * (dx_att / dist_to_offset)
                f_att_y = self.k_att * (dy_att / dist_to_offset)
            else:
                f_att_x = self.k_att * dx_att
                f_att_y = self.k_att * dy_att

            # Multi-Obstacle Repulsion Loop
            f_rep_x = f_rep_y = 0.0
            for opp in self.opponents_list:
                dx_obs = self.robot_x - opp['x']
                dy_obs = self.robot_y - opp['y']
                dist_to_obs = max(math.hypot(dx_obs, dy_obs), 0.001)

                if dist_to_obs < self.rho_0:
                    rad_x = dx_obs / dist_to_obs
                    rad_y = dy_obs / dist_to_obs
                    
                    tan_ax =  rad_y;  tan_ay = -rad_x
                    tan_bx = -rad_y;  tan_by =  rad_x
                    dot_a  = tan_ax * f_att_x + tan_ay * f_att_y
                    dot_b  = tan_bx * f_att_x + tan_by * f_att_y
                    
                    tan_x, tan_y = (tan_ax, tan_ay) if dot_a >= dot_b else (tan_bx, tan_by)
                    rep_mag = self.k_rep * (1.0 / dist_to_obs - 1.0 / self.rho_0) / (dist_to_obs ** 2)
                    w = self.tangent_weight
                    f_rep_x += rep_mag * ((1.0 - w) * rad_x + w * tan_x)
                    f_rep_y += rep_mag * ((1.0 - w) * rad_y + w * tan_y)

            global_fx = f_att_x + f_rep_x
            global_fy = f_att_y + f_rep_y
            target_heading = err_theta_ball
            dribbler_speed = 0.0

            # Stuck Escape Sequence
            if self._escape_ticks > 0:
                global_fx, global_fy = 0.0, self._escape_vy
                self._escape_ticks -= 1
            else:
                if self._callback_count % self.stuck_check_interval == 0:
                    if self._snap_x is not None:
                        moved = math.hypot(self.robot_x - self._snap_x, self.robot_y - self._snap_y)
                        if moved < self.stuck_dist_threshold and dist_to_offset > 0.15:
                            side = random.choice([-1.0, 1.0])
                            self._escape_vy    = side * self.escape_speed
                            self._escape_ticks = self.escape_duration
                    self._snap_x = self.robot_x
                    self._snap_y = self.robot_y

            # Transition: Position reached and face locked onto ball
            if dist_to_offset < 0.10 and abs(err_theta_ball) < 0.20 and self._escape_ticks == 0:
                self.state = FSMState.DRIBBLE_TO_GOAL
                self.get_logger().info("[STATE SWITCH] Ball captured. Dribbling toward opponent goal!")

        # ══════════════════════════════════════════════════════════════════════
        # State 1: DRIBBLE_TO_GOAL (Lock ball & Carry it forward while dodging)
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == FSMState.DRIBBLE_TO_GOAL:
            # Active dribbler ability enabled to capture and grip the ball
            dribbler_speed = 8.0  

            # Target is directly the center of opponent goal post
            dx_goal = goal_x - self.robot_x
            dy_goal = goal_y - self.robot_y
            dist_to_goal = max(math.hypot(dx_goal, dy_goal), 0.001)

            # Continuous Forward Attraction Force towards Goal
            f_att_goal_x = self.k_att * 1.2 * (dx_goal / dist_to_goal)
            f_att_goal_y = self.k_att * 1.2 * (dy_goal / dist_to_goal)

            # Keep APF evasion active WHILE dribbling so defenders don't snatch it
            f_rep_x = f_rep_y = 0.0
            for opp in self.opponents_list:
                dx_obs = self.robot_x - opp['x']
                dy_obs = self.robot_y - opp['y']
                dist_to_obs = max(math.hypot(dx_obs, dy_obs), 0.001)

                if dist_to_obs < self.rho_0:  # Obstacle avoidance safety ring
                    rad_x = dx_obs / dist_to_obs
                    rad_y = dy_obs / dist_to_obs
                    tan_ax =  rad_y;  tan_ay = -rad_x
                    tan_bx = -rad_y;  tan_by =  rad_x
                    dot_a  = tan_ax * f_att_goal_x + tan_ay * f_att_goal_y
                    dot_b  = tan_bx * f_att_goal_x + tan_by * f_att_goal_y
                    
                    tan_x, tan_y = (tan_ax, tan_ay) if dot_a >= dot_b else (tan_bx, tan_by)
                    rep_mag = (self.k_rep * 1.2) * (1.0 / dist_to_obs - 1.0 / self.rho_0) / (dist_to_obs ** 2)
                    f_rep_x += rep_mag * ((1.0 - self.tangent_weight) * rad_x + self.tangent_weight * tan_x)
                    f_rep_y += rep_mag * ((1.0 - self.tangent_weight) * rad_y + self.tangent_weight * tan_y)

            global_fx = f_att_goal_x + f_rep_x
            global_fy = f_att_goal_y + f_rep_y
            
            # Lock robot orientation strictly facing the opponent goal
            target_heading = err_theta_goal

            # Transition to Strike: Near penalty area/goal vicinity (~0.65 meters)
            dist_to_goal_zone = abs(goal_x - self.robot_x)
            if dist_to_goal_zone < 0.65 and abs(err_theta_goal) < 0.25:
                self.state = FSMState.STRIKE
                self.strike_tick_timer = 0
                self.get_logger().info("[STATE SWITCH] Goal within range! Launching kick strike!")

            # Recovery Fallback: Lost the ball during dribbling
            if dist_to_ball > 0.38:
                self.state = FSMState.GET_BEHIND_BALL
                self.get_logger().warn("[FALLBACK] Ball detached. Re-intercepting.")

        # ══════════════════════════════════════════════════════════════════════
        # State 2: STRIKE (Instantaneous Burst & High Impact Kick)
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == FSMState.STRIKE:
            # Charge toward goal using actual direction vector
            goal_dir_x = goal_x - self.robot_x
            goal_dir_y = goal_y - self.robot_y
            goal_dist  = max(math.hypot(goal_dir_x, goal_dir_y), 0.001)
            global_fx  = 4.0 * (goal_dir_x / goal_dist)
            global_fy  = 4.0 * (goal_dir_y / goal_dist)
            target_heading = err_theta_goal
            
            kick_speed     = 8.0  # Max permitted hardware striker punch speed
            dribbler_speed = 1.0  # Low dribbler for frictionless release physics

            self.strike_tick_timer += 1
            if self.strike_tick_timer > 10:  # ~0.3 seconds burst duration
                self.state = FSMState.GET_BEHIND_BALL
                self.get_logger().info("[KICK METRICS] Strike sequence completed. Resetting FSM.")

        # ── Global → robot-local frame (standard 2D rotation) ─────────────
        c, s = math.cos(self.robot_theta), math.sin(self.robot_theta)
        vx_cmd =  global_fx * c + global_fy * s
        vy_cmd = -global_fx * s + global_fy * c
        vw_cmd = self.kp_rot * target_heading

        # Profile Velocity Guardrails
        if self.state != FSMState.STRIKE:
            current_speed = math.hypot(vx_cmd, vy_cmd)
            if current_speed > self.max_linear_speed:
                scalar = self.max_linear_speed / current_speed
                vx_cmd *= scalar
                vy_cmd *= scalar
            vw_cmd = max(-self.max_angular_speed, min(self.max_angular_speed, vw_cmd))

        # Console Log Stream Updates
        if self._callback_count % 30 == 0:
            self.get_logger().info(
                f"Mode: {self.state_labels[self.state]} | Goal Distance: {abs(goal_x - self.robot_x):.2f}m | Tracked Obstacles: {len(self.opponents_list)}"
            )

        # Transmit direct UDP velocities
        try:
            pkt = build_udp_packet(self.robot_id, vx_cmd, vy_cmd, vw_cmd, kick_speed, dribbler_speed)
            self.sock.sendto(pkt, (self.grsim_host, self.grsim_port))
        except Exception as e:
            self.get_logger().error(f"UDP transmission failure: {e}")

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SSLAutonomousNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()