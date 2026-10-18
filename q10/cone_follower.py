import math
import csv
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


#find the nearest small object in the front LiDAR field of view
def find_nearest_cone(scan, fov_deg=90.0, max_width=0.5):
    fov = math.radians(fov_deg)
    clusters = []
    current = []
    previous_index = None
    for i, r in enumerate(scan.ranges):
        angle = scan.angle_min + i * scan.angle_increment
        angle = math.atan2(math.sin(angle), math.cos(angle))
        valid = (
            math.isfinite(r)
            and scan.range_min < r < scan.range_max
            and abs(angle) <= fov
        )
        if not valid:
            if current:
                clusters.append(current)
                current = []
            previous_index = None
            continue
        #split clusters at large range jumps or missing beam indices
        if current:
            previous_range = current[-1][0]
            gap = i- previous_index
            if gap!= 1 or abs(r -previous_range) > 0.15:
                clusters.append(current)
                current = []
        current.append((r, angle))
        previous_index = i

    if current:
        clusters.append(current)

    candidates = []
    for cluster in clusters:
        #estimate the physical width between the cluster endpoints
        r1, a1 = cluster[0]
        r2, a2 = cluster[-1]

        x1, y1 = r1 * math.cos(a1), r1 * math.sin(a1)
        x2, y2 = r2 * math.cos(a2), r2 * math.sin(a2)

        width = math.hypot(x2 - x1, y2 - y1)

        #reject long objects such as walls
        if width <= max_width:
            # Use the closest return within this cluster.
            rho, phi = min(cluster, key=lambda point: point[0])
            candidates.append((rho, phi))

    if not candidates:
        return None
    return min(candidates, key=lambda point: point[0])


class ConeFollower(Node):
    def __init__(self):
        super().__init__('cone_follower')

        self.kv = self.declare_parameter('Kv', 1.0).value
        self.kw = self.declare_parameter('Kw', 2.0).value
        self.fov_deg = self.declare_parameter('fov_deg', 90.0).value
        self.max_width = self.declare_parameter('max_width', 0.5).value
        self.tag = self.declare_parameter('tag', '').value
        
        self.gt = {'r': (math.nan,) * 3, 'c': (math.nan,) * 3}
        self.create_subscription(Odometry, '/ground_truth/odom', lambda m: self.set_gt('r', m), 10)
        self.create_subscription(Odometry, '/cone/ground_truth/odom', lambda m: self.set_gt('c', m), 10)

        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data
        )

        filename = f'cone_follower_Kv{self.kv}{self.tag}.csv'
        self.log_file = open(filename, 'w', newline='')
        self.logger = csv.writer(self.log_file)

        self.logger.writerow(['time', 'rho', 'phi', 'v', 'omega',
                              'robot_x', 'robot_y', 'robot_yaw', 'cone_x', 'cone_y'])
        self.start_time = None

        self.get_logger().info(
            f'Cone follower started: Kv={self.kv}, Kw={self.kw}'
        )

    def set_gt(self, who, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.gt[who] = (p.x, p.y, 2 * math.atan2(q.z, q.w))

    def on_scan(self, scan):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            self.start_time = now

        target = find_nearest_cone(
            scan,
            fov_deg=self.fov_deg,
            max_width=self.max_width
        )

        cmd = Twist()
        rho = math.nan
        phi = math.nan

        if target is not None:
            rho, phi = target

            #proportional controller with the required limits
            cmd.linear.x = max(
                0.0, min(self.kv * rho, 0.5)
            )
            cmd.angular.z = max(
                -1.5, min(self.kw * phi, 1.5)
            )

        #no valid target: publish zero velocity to stop
        self.cmd_pub.publish(cmd)

        self.logger.writerow([
            f'{now - self.start_time:.3f}',
            f'{rho:.4f}',
            f'{phi:.4f}',
            f'{cmd.linear.x:.4f}',
            f'{cmd.angular.z:.4f}',
            *[f'{x:.4f}' for x in self.gt['r'] + self.gt['c'][:2]]
        ])
        self.log_file.flush()

    def destroy_node(self):
        self.cmd_pub.publish(Twist())
        self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ConeFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()