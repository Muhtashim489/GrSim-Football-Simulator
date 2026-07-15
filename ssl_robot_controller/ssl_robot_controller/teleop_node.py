#!/usr/bin/env python3
"""
Phase 1: Teleoperation and System Setup Node.
Sends robot commands directly to grSim via UDP protobuf (SSL standard).
Uses Linux termios for non-blocking keyboard reads.
"""

import sys
import threading
import tty
import termios
import select
import socket
import struct
import rclpy
from rclpy.node import Node

# grSim robot control ports (SSL standard)
GRSIM_HOST = '127.0.0.1'
GRSIM_BLUE_PORT   = 10301
GRSIM_YELLOW_PORT = 10302

# Try to import protobuf generated classes
try:
    from ssl_league_protobufs import ssl_simulation_robot_control_pb2 as robot_ctrl_pb
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False


def get_key_linux(timeout=0.1):
    """Non-blocking key read on Linux using termios."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def build_robot_control_packet(robot_id, vx, vy, vw):
    """
    Build a protobuf RobotControl packet for grSim.
    Uses MoveLocalVelocity: forward=vx, left=vy, angular=vw.
    Falls back to a raw struct if protobuf is unavailable.
    """
    if PROTOBUF_AVAILABLE:
        control = robot_ctrl_pb.RobotControl()
        cmd = control.robot_commands.add()
        cmd.id = robot_id
        cmd.move_command.local_velocity.forward = vx
        cmd.move_command.local_velocity.left    = vy
        cmd.move_command.local_velocity.angular = vw
        cmd.kick_speed    = 0.0
        cmd.kick_angle    = 0.0
        cmd.dribbler_speed = 0.0
        return control.SerializeToString()
    else:
        # Minimal hand-crafted protobuf encoding as fallback
        # Field 1 (robot_commands, field_number=1, wire_type=2/LEN)
        # Inner message fields: id(1,VARINT), move_command(2,LEN)->local_velocity(2,LEN)->forward(1,fixed32),left(2,fixed32),angular(3,fixed32)
        def encode_varint(value):
            bits = value & 0x7f
            value >>= 7
            result = b''
            while value:
                result += bytes([0x80 | bits])
                bits = value & 0x7f
                value >>= 7
            result += bytes([bits])
            return result

        def encode_float(tag, value):
            return encode_varint(tag << 3 | 5) + struct.pack('<f', value)

        # local_velocity sub-message: forward(1), left(2), angular(3)
        local_vel = (encode_float(1, vx) +
                     encode_float(2, vy) +
                     encode_float(3, vw))

        # move_command sub-message: local_velocity is field 2
        move_cmd_inner = encode_varint(2 << 3 | 2) + encode_varint(len(local_vel)) + local_vel
        # robot_command sub-message: id(1,VARINT), move_command(2,LEN)
        robot_cmd = (encode_varint(1 << 3 | 0) + encode_varint(robot_id) +
                     encode_varint(2 << 3 | 2) + encode_varint(len(move_cmd_inner)) + move_cmd_inner)
        # RobotControl: robot_commands field 1, LEN
        packet = encode_varint(1 << 3 | 2) + encode_varint(len(robot_cmd)) + robot_cmd
        return packet


class SSLTeleopNode(Node):
    def __init__(self):
        super().__init__('ssl_teleop_node')

        # Declare parameters
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('is_team_yellow', False)
        self.declare_parameter('grsim_host', GRSIM_HOST)

        self.robot_id      = self.get_parameter('robot_id').value
        self.is_team_yellow = self.get_parameter('is_team_yellow').value
        self.grsim_host    = self.get_parameter('grsim_host').value
        self.grsim_port    = GRSIM_YELLOW_PORT if self.is_team_yellow else GRSIM_BLUE_PORT

        # UDP socket to grSim
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.get_logger().info(
            f"SSL Teleop Node Initialized.\n"
            f"Robot ID: {self.robot_id} | Team Yellow: {self.is_team_yellow}\n"
            f"Sending UDP to {self.grsim_host}:{self.grsim_port}\n"
            f"Protobuf library: {'available' if PROTOBUF_AVAILABLE else 'NOT found - using fallback encoder'}"
        )

        # Current velocities
        self.vx = 0.0
        self.vy = 0.0
        self.vw = 0.0

        self.speed_step = 0.5   # m/s
        self.yaw_step   = 1.0   # rad/s

        # Keyboard thread
        self.running = True
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        # 50 Hz publish timer
        self.timer = self.create_timer(0.02, self.publish_commands)

    def print_usage(self):
        print("""
---------------------------------------------
RoboCup SSL Holonomic Keyboard Teleoperation
---------------------------------------------
  W           : Move Forward  (vx +)
  S           : Move Backward (vx -)
  A           : Strafe Left   (vy +)
  D           : Strafe Right  (vy -)
  Q           : Spin Left     (vw +)
  E           : Spin Right    (vw -)
  SPACE       : Stop immediately
  C           : Print current speeds
  X / Ctrl+C  : Exit
---------------------------------------------
""")

    def keyboard_loop(self):
        self.print_usage()
        while self.running and rclpy.ok():
            key = get_key_linux(timeout=0.1)
            if key is None:
                continue

            key = key.lower()

            if key == 'w':
                self.vx = self.speed_step
                self.vy = 0.0
            elif key == 's':
                self.vx = -self.speed_step
                self.vy = 0.0
            elif key == 'a':
                self.vx = 0.0
                self.vy = self.speed_step
            elif key == 'd':
                self.vx = 0.0
                self.vy = -self.speed_step
            elif key == 'q':
                self.vw = self.yaw_step
            elif key == 'e':
                self.vw = -self.yaw_step
            elif key == ' ':
                self.vx = self.vy = self.vw = 0.0
                print("STOP")
            elif key == 'c':
                print(f"vx={self.vx:.2f} m/s  vy={self.vy:.2f} m/s  vw={self.vw:.2f} rad/s")
            elif key == 'x':
                self.vx = self.vy = self.vw = 0.0
                self.running = False
                self.get_logger().info("Exiting teleoperation...")
                break

            if key in ['w', 'a', 's', 'd', 'q', 'e']:
                print(f"CMD -> vx={self.vx:.2f}  vy={self.vy:.2f}  vw={self.vw:.2f}")

    def publish_commands(self):
        if not self.running:
            return
        try:
            packet = build_robot_control_packet(self.robot_id, self.vx, self.vy, self.vw)
            self.sock.sendto(packet, (self.grsim_host, self.grsim_port))
        except Exception as e:
            self.get_logger().error(f"UDP send failed: {e}")

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SSLTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()