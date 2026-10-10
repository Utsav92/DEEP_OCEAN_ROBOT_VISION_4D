"""ROS 2 sonar subscriber (MODE 6, optional). NEVER RUN: ROS 2 / rclpy are not installed in the development environment.

The pure conversion helpers below are unit-tested; the Node class is only defined when rclpy imports.
Topic conventions are configurable because there is no single standard multibeam message: this bridge accepts
(a) `sensor_msgs/LaserScan`-like fans (ranges + angle_min/angle_increment) and
(b) `sensor_msgs/Image` mono8 range-bearing imagery for forward-looking sonar.
"""
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    HAVE_RCLPY = True
except Exception:
    HAVE_RCLPY = False


def laserscan_to_ping(ranges, angle_min, angle_increment, stamp_s, range_max=260.0):
    """LaserScan-like fan -> (t, ranges (B,), beam_theta (B,) rad, valid (B,)). Angles are measured from nadir (+ starboard);
    `ranges` that are inf/NaN or beyond range_max are invalid. Values are calibrated ranges: no sound-speed conversion."""
    r = np.asarray(ranges, dtype=np.float64)
    th = angle_min + angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r > 0) & (r <= range_max)
    return float(stamp_s), np.where(ok, r, np.nan), th, ok


def image_to_polar(data, height, width, encoding="mono8"):
    if encoding != "mono8":
        raise ValueError("only mono8 imaging-sonar images are supported")
    return np.frombuffer(bytes(data), dtype=np.uint8).reshape(height, width)


if HAVE_RCLPY:
    class SonarBridge(Node):
        def __init__(self, topic="/sonar/scan", sink=None):
            super().__init__("dorv_sonar_bridge")
            from sensor_msgs.msg import LaserScan
            self.sink = sink or (lambda *a: None)
            self.create_subscription(LaserScan, topic, self.cb, 10)

        def cb(self, m):
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            self.sink(laserscan_to_ping(m.ranges, m.angle_min, m.angle_increment, t))
