#!/usr/bin/env python3
"""
Phase 2: Ball Tracking (Homing Behavior).

Subscribes to /ssl_vision_bridge/vision_messages (VisionWrapper).
Computes:
  - Distance to a target point behind the ball
  - Heading angle error (robot must face the ball)
Sends MoveLocalVelocity protobuf packets directly to grSim via UDP.

All coordinates from ssl_ros_bridge are already in meters.
Robot orientation arrives as a quaternion — converted to yaw here.
"""

import math
import socket
import struct
import rclpy
from rclpy.node import Node
from ssl_league_msgs.msg import VisionWrapper

# ── grSim UDP ports ────────────────────────────────────────────────────────────
GRSIM_HOST        = '127.0.0.1'
GRSIM_BLUE_PORT   = 10301
GRSIM_YELLOW_PORT = 10302

# ── Try compiled protobuf bindings (optional) ──────────────────────────────────
try:
    from ssl_league_protobufs import ssl_simulation_robot_control_pb2 as robot_ctrl_pb
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def quat_to_yaw(qx, qy, qz, qw):
    """Extract yaw (rotation around Z) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle >  math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle


def build_udp_packet(robot_id, vx, vy, vw):
    """
    Serialize a RobotControl protobuf.
    Uses MoveLocalVelocity: forward=vx, left=vy, angular=vw.
    Falls back to hand-crafted encoding when compiled bindings are absent.
    """
    if PROTOBUF_AVAILABLE:
        ctrl = robot_ctrl_pb.RobotControl()
        cmd  = ctrl.robot_commands.add()
        cmd.id = robot_id
        cmd.move_command.local_velocity.forward = vx
        cmd.move_command.local_velocity.left    = vy
        cmd.move_command.local_velocity.angular = vw
        cmd.kick_speed     = 0.0
        cmd.kick_angle     = 0.0
        cmd.dribbler_speed = 0.0
        return ctrl.SerializeToString()

    # ── Hand-crafted protobuf fallback ────────────────────────────────────────
    def varint(v):
        out = b''
        while True:
            bits = v & 0x7F
            v >>= 7
            out += bytes([bits | (0x80 if v else 0)])
            if not v:
                break
        return out

    def pfloat(tag, val):
        return varint(tag << 3 | 5) + struct.pack('<f', val)

    local_vel    = pfloat(1, vx) + pfloat(2, vy) + pfloat(3, vw)
    move_inner   = varint(2 << 3 | 2) + varint(len(local_vel))   + local_vel
    robot_cmd    = (varint(1 << 3 | 0) + varint(robot_id) +
                    varint(2 << 3 | 2) + varint(len(move_inner)) + move_inner)
    packet       = varint(1 << 3 | 2) + varint(len(robot_cmd)) + robot_cmd
    return packet


# ══════════════════════════════════════════════════════════════════════════════
# Node
# ══════════════════════════════════════════════════════════════════════════════

class SSLTrackerNode(Node):

    def __init__(self):
        super().__init__('ssl_tracker_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('robot_id',         0)
        self.declare_parameter('is_team_yellow',   False)
        self.declare_parameter('opponent_goal_x',  4.5)   # m — right goal
        self.declare_parameter('offset_distance',  0.3)   # m — behind the ball
        self.declare_parameter('grsim_host',       GRSIM_HOST)

        self.robot_id         = self.get_parameter('robot_id').value
        self.is_team_yellow   = self.get_parameter('is_team_yellow').value
        self.opponent_goal_x  = self.get_parameter('opponent_goal_x').value
        self.offset_distance  = self.get_parameter('offset_distance').value
        self.grsim_host       = self.get_parameter('grsim_host').value
        self.grsim_port       = GRSIM_YELLOW_PORT if self.is_team_yellow else GRSIM_BLUE_PORT

        # ── Proportional controller gains ─────────────────────────────────────
        self.kp_trans         = 2.0   # translational gain
        self.kp_rot           = 4.0   # rotational gain
        self.max_linear_speed = 1.5   # m/s
        self.max_angular_speed= 3.0   # rad/s

        # Arrival thresholds (deadband to stop jitter)
        self.pos_deadband     = 0.05  # m   — stop translating when this close
        self.rot_deadband     = 0.05  # rad — stop rotating  when this aligned

        # ── State ─────────────────────────────────────────────────────────────
        self.robot_x     = None
        self.robot_y     = None
        self.robot_theta = None   # yaw in radians
        self.ball_x      = None
        self.ball_y      = None

        # ── UDP socket ────────────────────────────────────────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ── ROS subscriber ────────────────────────────────────────────────────
        # The bridge node publishes on ~/vision_messages  →  /ssl_vision_bridge/vision_messages
        self.vision_sub = self.create_subscription(
            VisionWrapper,
            '/ssl_vision_bridge/vision_messages',
            self.vision_callback,
            10
        )

        self.get_logger().info(
            f"Tracker Node started.\n"
            f"  Robot ID      : {self.robot_id}\n"
            f"  Team Yellow   : {self.is_team_yellow}\n"
            f"  UDP target    : {self.grsim_host}:{self.grsim_port}\n"
            f"  Offset dist   : {self.offset_distance} m\n"
            f"  Protobuf lib  : {'OK' if PROTOBUF_AVAILABLE else 'missing — using fallback'}\n"
            f"Waiting for vision messages …"
        )

    # ── Vision callback ───────────────────────────────────────────────────────

    def vision_callback(self, msg: VisionWrapper):
        """
        VisionWrapper.detection  →  list[VisionDetectionFrame]
        Each frame:
          .balls[]       → VisionDetectionBall  (.pos.x, .pos.y  in metres)
          .robots_blue[] → VisionDetectionRobot (.pose.position.x/y, .pose.orientation)
          .robots_yellow[]
        """
        if not msg.detection:
            return   # no detection frame in this packet

        # Merge all camera frames: pick ball with highest confidence,
        # pick our robot from whichever frame it appears in.
        best_ball_conf  = -1.0
        best_robot_conf = -1.0

        for frame in msg.detection:
            # ── Ball ──────────────────────────────────────────────────────────
            for ball in frame.balls:
                if ball.confidence > best_ball_conf:
                    best_ball_conf = ball.confidence
                    self.ball_x    = ball.pos.x   # already metres
                    self.ball_y    = ball.pos.y

            # ── Our robot ─────────────────────────────────────────────────────
            robots = frame.robots_yellow if self.is_team_yellow else frame.robots_blue
            for robot in robots:
                if robot.robot_id == self.robot_id and robot.confidence > best_robot_conf:
                    best_robot_conf  = robot.confidence
                    self.robot_x     = robot.pose.position.x   # metres
                    self.robot_y     = robot.pose.position.y
                    q = robot.pose.orientation
                    self.robot_theta = quat_to_yaw(q.x, q.y, q.z, q.w)

        # Only act when both ball and robot are observed
        if (self.ball_x is None or
                self.robot_x is None):
            return

        self._control_loop()

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        """
        Compute a target point behind the ball (away from the opponent goal),
        then run a proportional controller in the robot's local frame.

        Perception-to-motion mapping (Phase 2 objective):
          1. Compute distance & heading from robot to target
          2. Convert global error to robot-local frame
          3. Scale to velocity commands
          4. Saturate & apply deadband
          5. Send via UDP
        """

        # ── 1. Target point: offset behind the ball ───────────────────────────
        #   Direction vector from opponent goal  →  ball
        goal_x = self.opponent_goal_x
        dx_g2b = self.ball_x - goal_x
        dy_g2b = self.ball_y - 0.0
        dist_g2b = math.hypot(dx_g2b, dy_g2b)

        if dist_g2b < 0.01:
            # Ball is essentially at the goal centre — nothing safe to do
            return

        ux = dx_g2b / dist_g2b   # unit vector goal → ball
        uy = dy_g2b / dist_g2b

        # Place target offset behind ball (opposite side from goal)
        target_x = self.ball_x + self.offset_distance * ux
        target_y = self.ball_y + self.offset_distance * uy

        # ── 2. Global position error ──────────────────────────────────────────
        err_gx = target_x - self.robot_x
        err_gy = target_y - self.robot_y
        dist_to_target = math.hypot(err_gx, err_gy)

        # ── 3. Transform error into robot-local frame ─────────────────────────
        #   Local X = forward,  Local Y = left
        cos_t = math.cos(self.robot_theta)
        sin_t = math.sin(self.robot_theta)
        err_local_x =  err_gx * cos_t + err_gy * sin_t
        err_local_y = -err_gx * sin_t + err_gy * cos_t

        # ── 4. Heading error (robot should face the ball) ─────────────────────
        desired_heading = math.atan2(
            self.ball_y - self.robot_y,
            self.ball_x - self.robot_x
        )
        err_theta = normalize_angle(desired_heading - self.robot_theta)

        # ── 5. Proportional commands ──────────────────────────────────────────
        vx_cmd = self.kp_trans * err_local_x
        vy_cmd = self.kp_trans * err_local_y
        vw_cmd = self.kp_rot   * err_theta

        # ── 6. Saturate linear speed ──────────────────────────────────────────
        speed = math.hypot(vx_cmd, vy_cmd)
        if speed > self.max_linear_speed:
            scale  = self.max_linear_speed / speed
            vx_cmd *= scale
            vy_cmd *= scale

        vw_cmd = max(-self.max_angular_speed,
                 min( self.max_angular_speed, vw_cmd))

        # ── 7. Deadband ───────────────────────────────────────────────────────
        if dist_to_target < self.pos_deadband:
            vx_cmd = 0.0
            vy_cmd = 0.0

        if abs(err_theta) < self.rot_deadband:
            vw_cmd = 0.0

        # ── 8. Log (throttled) ────────────────────────────────────────────────
        self.get_logger().info(
            f"ball=({self.ball_x:.2f},{self.ball_y:.2f})  "
            f"robot=({self.robot_x:.2f},{self.robot_y:.2f})  "
            f"θ={math.degrees(self.robot_theta):.1f}°  "
            f"dist={dist_to_target:.2f}m  "
            f"cmd=({vx_cmd:.2f},{vy_cmd:.2f},{vw_cmd:.2f})",
            throttle_duration_sec=0.5
        )

        # ── 9. Send UDP packet ────────────────────────────────────────────────
        try:
            packet = build_udp_packet(self.robot_id, vx_cmd, vy_cmd, vw_cmd)
            self.sock.sendto(packet, (self.grsim_host, self.grsim_port))
        except Exception as e:
            self.get_logger().error(f"UDP send failed: {e}")

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = SSLTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()