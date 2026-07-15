#!/usr/bin/env python3
"""
Phase 4: Autonomous Strike Node (Terminal Parameter Controlled Dynamic Obstacles)
Behavior Flow:
  1. GET_BEHIND_BALL: Navigates behind the ball using APF obstacle avoidance.
  2. DRIBBLE_TO_GOAL: Drives the ball towards the opponent's goal while actively 
     dodging MOVING opponent robots configured live from the terminal.
  3. STRIKE: Executes a high-speed kick. If blocked by a moving defender,
     instantly loops back to try again continuously.
"""

import sys
import math
import random
import socket
import struct
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
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

        # ── Terminal Dynamically Modifiable Parameters ────────────────────────
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('is_team_yellow', False)  # Striker ki apni team
        self.declare_parameter('moving_opponent_ids', [0])  # Kaunse opponents move karenge
        self.declare_parameter('opponent_team_yellow', True)  # Opponents ki team kaunsi hai
        
        self.declare_parameter('offset_distance', 0.25) 
        self.declare_parameter('vision_topic', '/ssl_vision_bridge/vision_messages')
        self.declare_parameter('grsim_host', GRSIM_HOST)

        # Parameter values load karein
        self.load_parameters()

        # Add dynamically adjustable callback listener from terminal
        self.add_on_set_parameters_callback(self.parameters_callback)

        # ── APF Gains ─────────────────────────────────────────────────────────
        self.k_att           = 4.6   
        self.k_rep           = 8.5   
        self.rho_0           = 0.85  
        self.d_switch        = 0.45
        self.tangent_weight  = 0.85  

        self.max_linear_speed  = 2.3
        self.max_angular_speed = 5.5
        self.kp_rot            = 7.0

        # Stuck Recovery System
        self.stuck_check_interval  = 30
        self.stuck_dist_threshold  = 0.04
        self.escape_duration       = 15
        self.escape_speed          = 0.9
        self._callback_count  = 0
        self._snap_x          = None
        self._snap_y          = None
        self._escape_ticks    = 0
        self._escape_vy       = 0.0

        # FSM Configuration
        self.state = FSMState.GET_BEHIND_BALL
        self.state_labels = {
            FSMState.GET_BEHIND_BALL: "GET_BEHIND_BALL",
            FSMState.DRIBBLE_TO_GOAL: "DRIBBLE_TO_GOAL",
            FSMState.STRIKE: "STRIKE"
        }
        self.strike_tick_timer = 0

        # Data Track Storage
        self.robot_x = self.robot_y = self.robot_theta = None
        self.ball_x  = self.ball_y  = None
        self.opponents_list = []

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.vision_sub = self.create_subscription(
            VisionWrapper,
            self.vision_topic_name,
            self.vision_callback, 10
        )

        self.print_status_report()

    def load_parameters(self):
        self.robot_id             = self.get_parameter('robot_id').value
        self.is_team_yellow       = self.get_parameter('is_team_yellow').value
        self.moving_opponent_ids  = self.get_parameter('moving_opponent_ids').value
        self.opponent_team_yellow = self.get_parameter('opponent_team_yellow').value
        self.offset_distance      = self.get_parameter('offset_distance').value
        self.vision_topic_name    = self.get_parameter('vision_topic').value
        self.grsim_host           = self.get_parameter('grsim_host').value

        # Network ports routing based on values
        self.grsim_port          = GRSIM_YELLOW_PORT if self.is_team_yellow else GRSIM_BLUE_PORT
        self.opponent_grsim_port = GRSIM_YELLOW_PORT if self.opponent_team_yellow else GRSIM_BLUE_PORT
        self.opponent_goal_x     = -4.5 if self.is_team_yellow else 4.5

    def parameters_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        result = SetParametersResult()
        result.successful = True
        
        # Reload parameters on terminal inputs dynamically
        for param in params:
            if param.name == 'robot_id':
                self.robot_id = param.value
            elif param.name == 'is_team_yellow':
                self.is_team_yellow = param.value
            elif param.name == 'moving_opponent_ids':
                self.moving_opponent_ids = param.value
            elif param.name == 'opponent_team_yellow':
                self.opponent_team_yellow = param.value
                
        # Recalculate side targets based on live changes
        self.grsim_port          = GRSIM_YELLOW_PORT if self.is_team_yellow else GRSIM_BLUE_PORT
        self.opponent_grsim_port = GRSIM_YELLOW_PORT if self.opponent_team_yellow else GRSIM_BLUE_PORT
        self.opponent_goal_x     = -4.5 if self.is_team_yellow else 4.5
        
        self.print_status_report()
        return result

    def print_status_report(self):
        self.get_logger().info(
            f"\n=== LIVE TERMINAL CONFIGURATION APPLIED ===\n"
            f"  Striker Agent ID : {self.robot_id} | Team: {'YELLOW' if self.is_team_yellow else 'BLUE'}\n"
            f"  Target Goal X    : {self.opponent_goal_x}m\n"
            f"  Opponent Team    : {'YELLOW' if self.opponent_team_yellow else 'BLUE'}\n"
            f"  Moving Opponents : {self.moving_opponent_ids}\n"
            f"============================================"
        )

    def vision_callback(self, msg: VisionWrapper):
        if not msg.detection:
            return

        best_ball = best_striker = -1.0
        opponents_candidate = []

        for frame in msg.detection:
            # Ball Tracking
            for ball in frame.balls:
                if ball.confidence > best_ball:
                    best_ball   = ball.confidence
                    self.ball_x = ball.pos.x
                    self.ball_y = ball.pos.y

            # Striker Pose Tracking
            my_team_robots = frame.robots_yellow if self.is_team_yellow else frame.robots_blue
            for r in my_team_robots:
                if r.robot_id == self.robot_id and r.confidence > best_striker:
                    best_striker     = r.confidence
                    self.robot_x     = r.pose.position.x
                    self.robot_y     = r.pose.position.y
                    q                = r.pose.orientation
                    self.robot_theta = quat_to_yaw(q.x, q.y, q.z, q.w)

            # Opponent Detection Tracking based on chosen configuration
            opp_team_label = "yellow" if self.opponent_team_yellow else "blue"
            opp_robots = frame.robots_yellow if self.opponent_team_yellow else frame.robots_blue
            
            for r in opp_robots:
                if r.confidence > 0.4:
                    opponents_candidate.append({
                        'unique_key': f"{opp_team_label}_{r.robot_id}",
                        'id': r.robot_id,
                        'x': r.pose.position.x,
                        'y': r.pose.position.y
                    })

        seen_keys = set()
        self.opponents_list = []
        for opp in opponents_candidate:
            if opp['unique_key'] not in seen_keys:
                seen_keys.add(opp['unique_key'])
                self.opponents_list.append(opp)

        if self.ball_x is not None and self.robot_x is not None and self.robot_theta is not None:
            self._callback_count += 1
            
            # ── DRIVE SELECTED OPPONENTS FROM TERMINAL ────────────────────────
            patrol_velocity = 1.3 * math.sin(self._callback_count * 0.04)
            for opp_id in self.moving_opponent_ids:
                try:
                    opp_pkt = build_udp_packet(opp_id, 0.0, patrol_velocity, 0.0, 0.0, 0.0)
                    self.sock.sendto(opp_pkt, (self.grsim_host, self.opponent_grsim_port))
                except Exception:
                    pass

            self.process_state_machine()

    def process_state_machine(self):
        goal_x = self.opponent_goal_x
        goal_y = 0.0

        # APF geometry vectors
        dx_g2b = self.ball_x - goal_x
        dy_g2b = self.ball_y - goal_y
        dist_g2b = max(math.hypot(dx_g2b, dy_g2b), 0.001)
        ux, uy = dx_g2b / dist_g2b, dy_g2b / dist_g2b

        target_x = self.ball_x + self.offset_distance * ux
        target_y = self.ball_y + self.offset_distance * uy

        dist_to_offset = math.hypot(target_x - self.robot_x, target_y - self.robot_y)
        dist_to_ball   = math.hypot(self.ball_x - self.robot_x, self.ball_y - self.robot_y)

        angle_to_ball  = math.atan2(self.ball_y - self.robot_y, self.ball_x - self.robot_x)
        err_theta_ball = normalize_angle(angle_to_ball - self.robot_theta)

        angle_to_goal  = math.atan2(goal_y - self.robot_y, goal_x - self.robot_x)
        err_theta_goal = normalize_angle(angle_to_goal - self.robot_theta)

        global_fx, global_fy = 0.0, 0.0
        target_heading = err_theta_ball
        kick_speed = 0.0
        dribbler_speed = 0.0

        # ══════════════════════════════════════════════════════════════════════
        # State 0: GET_BEHIND_BALL
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

            f_rep_x = f_rep_y = 0.0
            for opp in self.opponents_list:
                dx_obs = self.robot_x - opp['x']
                dy_obs = self.robot_y - opp['y']
                dist_to_obs = max(math.hypot(dx_obs, dy_obs), 0.001)

                if dist_to_obs < self.rho_0:
                    rad_x = dx_obs / dist_to_obs
                    rad_y = dy_obs / dist_to_obs
                    tan_ax, tan_ay = rad_y, -rad_x
                    tan_bx, tan_by = -rad_y, rad_x
                    dot_a = tan_ax * f_att_x + tan_ay * f_att_y
                    dot_b = tan_bx * f_att_x + tan_by * f_att_y
                    
                    tan_x, tan_y = (tan_ax, tan_ay) if dot_a >= dot_b else (tan_bx, tan_by)
                    rep_mag = self.k_rep * (1.0 / dist_to_obs - 1.0 / self.rho_0) / (dist_to_obs ** 2)
                    f_rep_x += rep_mag * ((1.0 - self.tangent_weight) * rad_x + self.tangent_weight * tan_x)
                    f_rep_y += rep_mag * ((1.0 - self.tangent_weight) * rad_y + self.tangent_weight * tan_y)

            global_fx = f_att_x + f_rep_x
            global_fy = f_att_y + f_rep_y
            target_heading = err_theta_ball

            # Recovery module
            if self._escape_ticks > 0:
                global_fx, global_fy = 0.0, self._escape_vy
                self._escape_ticks -= 1
            else:
                if self._callback_count % self.stuck_check_interval == 0:
                    if self._snap_x is not None:
                        moved = math.hypot(self.robot_x - self._snap_x, self.robot_y - self._snap_y)
                        if moved < self.stuck_dist_threshold and dist_to_offset > 0.15:
                            self._escape_vy    = random.choice([-1.0, 1.0]) * self.escape_speed
                            self._escape_ticks = self.escape_duration
                    self._snap_x = self.robot_x
                    self._snap_y = self.robot_y

            if dist_to_offset < 0.12 and abs(err_theta_ball) < 0.22 and self._escape_ticks == 0:
                self.state = FSMState.DRIBBLE_TO_GOAL

        # ══════════════════════════════════════════════════════════════════════
        # State 1: DRIBBLE_TO_GOAL
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == FSMState.DRIBBLE_TO_GOAL:
            dribbler_speed = 8.0 

            dx_goal = goal_x - self.robot_x
            dy_goal = goal_y - self.robot_y
            dist_to_goal = max(math.hypot(dx_goal, dy_goal), 0.001)

            f_att_goal_x = self.k_att * 1.6 * (dx_goal / dist_to_goal)
            f_att_goal_y = self.k_att * 1.6 * (dy_goal / dist_to_goal)

            f_rep_x = f_rep_y = 0.0
            for opp in self.opponents_list:
                dx_obs = self.robot_x - opp['x']
                dy_obs = self.robot_y - opp['y']
                dist_to_obs = max(math.hypot(dx_obs, dy_obs), 0.001)

                if dist_to_obs < self.rho_0: 
                    rad_x = dx_obs / dist_to_obs
                    rad_y = dy_obs / dist_to_obs
                    tan_ax, tan_ay = rad_y, -rad_x
                    tan_bx, tan_by = -rad_y, rad_x
                    dot_a = tan_ax * f_att_goal_x + tan_ay * f_att_goal_y
                    dot_b = tan_bx * f_att_goal_x + tan_by * f_att_goal_y
                    
                    tan_x, tan_y = (tan_ax, tan_ay) if dot_a >= dot_b else (tan_bx, tan_by)
                    rep_mag = (self.k_rep * 1.8) * (1.0 / dist_to_obs - 1.0 / self.rho_0) / (dist_to_obs ** 2)
                    f_rep_x += rep_mag * ((1.0 - self.tangent_weight) * rad_x + self.tangent_weight * tan_x)
                    f_rep_y += rep_mag * ((1.0 - self.tangent_weight) * rad_y + self.tangent_weight * tan_y)

            global_fx = f_att_goal_x + f_rep_x
            global_fy = f_att_goal_y + f_rep_y
            target_heading = err_theta_goal

            if abs(goal_x - self.robot_x) < 0.85 and abs(err_theta_goal) < 0.25:
                self.state = FSMState.STRIKE
                self.strike_tick_timer = 0

            if dist_to_ball > 0.35:
                self.state = FSMState.GET_BEHIND_BALL

        # ══════════════════════════════════════════════════════════════════════
        # State 2: STRIKE
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == FSMState.STRIKE:
            global_fx = 4.5 if goal_x > 0 else -4.5
            global_fy = 0.0
            target_heading = err_theta_goal
            kick_speed, dribbler_speed = 8.0, 1.0  

            if dist_to_ball > 0.42:
                self.state = FSMState.GET_BEHIND_BALL

            self.strike_tick_timer += 1
            if self.strike_tick_timer > 12: 
                self.state = FSMState.GET_BEHIND_BALL

        # ── EXACT REFERENCE FRAME MATRIX TRANSFORM ────────────────────────────
        c, s = math.cos(-self.robot_theta), math.sin(-self.robot_theta)
        vx_cmd = global_fx * c - global_fy * s
        vy_cmd = global_fx * s + global_fy * c
        vw_cmd = self.kp_rot * target_heading

        if self.state != FSMState.STRIKE:
            current_speed = math.hypot(vx_cmd, vy_cmd)
            if current_speed > self.max_linear_speed:
                scalar = self.max_linear_speed / current_speed
                vx_cmd *= scalar
                vy_cmd *= scalar
            vw_cmd = max(-self.max_angular_speed, min(self.max_angular_speed, vw_cmd))

        try:
            pkt = build_udp_packet(self.robot_id, vx_cmd, vy_cmd, vw_cmd, kick_speed, dribbler_speed)
            self.sock.sendto(pkt, (self.grsim_host, self.grsim_port))
        except Exception as e:
            self.get_logger().error(f"UDP Link fail: {e}")

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