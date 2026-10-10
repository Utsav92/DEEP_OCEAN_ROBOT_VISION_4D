"""ROS 2 navigation subscriber (MODE 6, optional; NEVER RUN - no rclpy here). Pure helpers are unit-tested."""
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    HAVE_RCLPY = True
except Exception:
    HAVE_RCLPY = False


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def odom_to_pose_enu(px, py, pz, qx, qy, qz, qw, frame="ned"):
    """nav_msgs/Odometry pose -> 4x4 T_enu_vehicle. frame 'ned' (north-east-down world, FRD body, the usual marine
    convention) or 'enu' (REP-103). The conversion to this project's ENU/FRD frames is explicit and tested."""
    from preprocessing.coordinate_systems import R_ENU_NED
    R = quat_to_R(qx, qy, qz, qw)
    T = np.eye(4)
    if frame == "ned":
        T[:3, :3] = R_ENU_NED @ R
        T[:3, 3] = R_ENU_NED @ np.array([px, py, pz])
    else:                                               # ENU world, FLU body -> FRD body
        flu_to_frd = np.diag([1.0, -1.0, -1.0])
        T[:3, :3] = R @ flu_to_frd
        T[:3, 3] = [px, py, pz]
    return T


if HAVE_RCLPY:
    class NavigationBridge(Node):
        def __init__(self, topic="/odom", sink=None, frame="ned"):
            super().__init__("dorv_navigation_bridge")
            from nav_msgs.msg import Odometry
            self.sink, self.frame = sink or (lambda *a: None), frame
            self.create_subscription(Odometry, topic, self.cb, 20)

        def cb(self, m):
            p, q = m.pose.pose.position, m.pose.pose.orientation
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            self.sink(t, odom_to_pose_enu(p.x, p.y, p.z, q.x, q.y, q.z, q.w, self.frame))
