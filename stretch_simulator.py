import math

import rclpy
import rclpy.time
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from rclpy.node import Node

from stretch_mppi.tf2_wrapper import TF2Wrapper
from stretch_mppi.vis_utils import VisualizationUtils


def _ros2_time_to_sec(t: rclpy.time.Time | TimeMsg) -> float:
    # why is ros2 like this :(
    if isinstance(t, TimeMsg):
        sec, nanosec = t.sec, t.nanosec
    elif isinstance(t, rclpy.time.Time):
        sec, nanosec = t.seconds_nanoseconds()
    else:
        raise ValueError()

    return float(sec) + float(nanosec) * 1e-9


def _normalize_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


class SimulatorNode(Node):
    def __init__(self) -> None:
        super().__init__("stretch_simulator")

        self._state = (0.0, 0.0, 0.0)
        """x, y, theta"""
        self._state_history = [self._state]

        self._latest_command: TwistStamped | None = None
        self._command_timeout = 0.5

        self._cmd_vel_sub = self.create_subscription(
            TwistStamped, "/stretch/cmd_vel", self._cmd_vel_callback, 1
        )
        self._rviz_initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._initialpose_callback, 1
        )

        self._tf2_wrapper = TF2Wrapper(self)
        self._vis_utils = VisualizationUtils(self)

        self._tick_duration = 1 / 30
        self._timer = self.create_timer(self._tick_duration, self.tick)

    def _cmd_vel_callback(self, msg: TwistStamped) -> None:
        self._latest_command = msg

    def _initialpose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._state = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            2.0 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
        )
        self._state_history = [self._state]

    def _get_latest_2d_twist(self) -> tuple[float, float]:
        if self._latest_command is None:
            return 0.0, 0.0

        msg_stamp = _ros2_time_to_sec(self._latest_command.header.stamp)
        current_stamp = _ros2_time_to_sec(self.get_clock().now())

        if msg_stamp < current_stamp - self._command_timeout:
            return 0.0, 0.0

        return self._latest_command.twist.linear.x, self._latest_command.twist.angular.z

    def _step_dynamics(self) -> tuple[int, int, int]:
        s = self._state
        v_x, w_z = self._get_latest_2d_twist()

        s2_ego = [0.0, 0.0, 0.0]
        if w_z == 0.0:
            s2_ego[0] = v_x * self._tick_duration
        else:
            s2_ego[2] = w_z * self._tick_duration
            turning_radius = v_x / w_z
            s2_ego[0] = turning_radius * math.sin(s2_ego[2])
            s2_ego[1] = turning_radius * (1 - math.cos(s2_ego[2]))

        s2_global = [0.0, 0.0, 0.0]
        s2_global[0] = math.cos(s[2]) * s2_ego[0] - math.sin(s[2]) * s2_ego[1] + s[0]
        s2_global[1] = math.sin(s[2]) * s2_ego[0] + math.cos(s[2]) * s2_ego[1] + s[1]
        s2_global[2] = _normalize_angle(s[2] + s2_ego[2])

        return tuple(s2_global)  # type: ignore

    def tick(self) -> None:
        self._state = self._step_dynamics()

        if (
            math.sqrt(
                (self._state[0] - self._state_history[-1][0]) ** 2
                + (self._state[1] - self._state_history[-1][1]) ** 2
            )
            > 0.05
        ):
            self._state_history.append(self._state)

        self._tf2_wrapper.publish_2d_pose(
            "map",
            "base_link",
            *self._state,
        )
        self._vis_utils.visualize_path(self._state_history)


if __name__ == "__main__":
    rclpy.init()
    node = SimulatorNode()
    rclpy.spin(node)
    rclpy.shutdown()
