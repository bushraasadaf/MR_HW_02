import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

import numpy as np
import math
import os
import csv

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        #parameters
        self.declare_parameter('Ld', 0.5)
        self.declare_parameter('v', 0.2)
        self.declare_parameter('path', 'circle')
        self.Ld = float(self.get_parameter('Ld').value)
        self.v = float(self.get_parameter('v').value)
        self.path_name = self.get_parameter('path').value
        self.w_max = 1.5
        path_file = os.path.join(
            get_package_share_directory('a2_description'),
            'paths',
            self.path_name + '.csv'
        )

        self.path = np.loadtxt(
            path_file,
            delimiter=',',
            skiprows=1
        )

        #current progress along path
        self.nearest_idx = 0

        self.true_x = None
        self.true_y = None
        self.true_yaw = None

        #time
        self.t0 = None
        self.finished = False

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
         #ground truth for evaluation
        self.gt_sub = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self.gt_callback,
            10
        )

        #csv
        self.declare_parameter('tag', '')
        self.tag = self.get_parameter('tag').value
        filename = f'pure_pursuit_{self.path_name}_Ld{self.Ld}{self.tag}.csv'

        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            'time', 'odom_x','odom_y', 'odom_yaw', 'true_x','true_y', 'true_yaw',
            'cte_odom','cte_true', 'omega'
        ])

        self.get_logger().info(
            f'Loaded {self.path_name} path, Ld={self.Ld}'
        )

    #convert quaternion to yaw
    def yaw_from_quat(self, q):
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return yaw

    #save ground-truth pose
    def gt_callback(self, msg):
        self.true_x = msg.pose.pose.position.x
        self.true_y = msg.pose.pose.position.y
        self.true_yaw = self.yaw_from_quat(
            msg.pose.pose.orientation
        )

    #cross-track error
    def cross_track(self, x, y):
        dx = self.path[:, 0] - x
        dy = self.path[:, 1] - y

        distances = np.hypot(dx, dy)
        i = int(np.argmin(distances))
        side = (-math.sin(self.path[i, 2]) * (x - self.path[i, 0])
                + math.cos(self.path[i, 2]) * (y - self.path[i, 1]))
        return math.copysign(distances[i], side)   #ect>0 (left), ect<0 (right)

    #pure-pursuit controller
    def odom_callback(self, msg):
        if self.finished:
            return
       #current odometry pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.yaw_from_quat(
            msg.pose.pose.orientation
        )

        #time
        t = self.get_clock().now().nanoseconds / 1e9

        if self.t0 is None:
            self.t0 = t

        #find nearest point (forward only)
        best_idx = self.nearest_idx
        best_dist = float('inf')

        for i in range(self.nearest_idx, len(self.path)):
            dx = self.path[i, 0] - x
            dy = self.path[i, 1] - y
            distance = math.hypot(dx, dy)
 
            if distance < best_dist:
                best_dist = distance
                best_idx = i

            #no need to search the entire path in this case:
            if i > self.nearest_idx + 150:
                break
        self.nearest_idx = best_idx

        #check if lap is finished
        if self.nearest_idx >= len(self.path) - 2:
            self.stop_robot()
            self.finished = True
            self.get_logger().info('Lap complete!')
            return

        #find lookahead goal
        goal_idx = len(self.path) - 1 #end of the path: use the final point as the goal
        for i in range(self.nearest_idx, len(self.path)):
            dx = self.path[i, 0] - x
            dy = self.path[i, 1] - y
            distance = math.hypot(dx, dy)
            if distance >= self.Ld:
                goal_idx = i
                break

        goal_x = self.path[goal_idx, 0]
        goal_y = self.path[goal_idx, 1]

        #transform goal into robot frame
        dx = goal_x - x
        dy = goal_y - y

        x_robot = (
            math.cos(yaw) * dx
            + math.sin(yaw) * dy
        )
        y_robot = (
            -math.sin(yaw) * dx
            + math.cos(yaw) * dy
        )

        #pure pursuit
        L2 = x_robot**2 + y_robot**2

        if L2 > 0.000001:
            curvature = 2.0 * y_robot / L2
        else:
            curvature = 0.0

        omega = self.v * curvature

        #limit angular velocity
        omega = max(
            -self.w_max,
            min(self.w_max, omega)
        )

        #publish command
        cmd = Twist()
        cmd.linear.x = self.v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

        #cross track errors
        cte_odom = self.cross_track(x, y)
        if self.true_x is not None:
            cte_true = self.cross_track(
                self.true_x,
                self.true_y
            )
        else:
            cte_true = float('nan')

        #log csv
        self.csv_writer.writerow([
            t - self.t0, x,y,yaw,self.true_x,self.true_y, self.true_yaw,
            cte_odom,cte_true, omega
        ])

        self.csv_file.flush()


    def stop_robot(self):
        if rclpy.ok():
            cmd = Twist()
            self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.stop_robot()
    node.csv_file.close()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()