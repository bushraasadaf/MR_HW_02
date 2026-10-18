import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

#converting quaternion to yaw
def yaw(m):
    q = m.pose.pose.orientation
    return 2 * math.atan2(q.z, q.w)
##wrap angle difference to [-pi, pi]
def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

class SpinTest(Node):
    def __init__(self):
        super().__init__('spin_test')
        self.w = self.declare_parameter('omega', 0.5).value
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        ##subscribe to estimated and ground truth odometry
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.on_gt, 10)
        #tracking variables for wrapped and unwrapped yaw
        self.o = self.g = None       #last wrapped yaw
        self.so = self.sg = 0.0      #unwrapped totals
        self.state, self.done = 'wait', False
        #control loop running at 50Hz:
        self.create_timer(0.02, self.step)

    def on_odom(self, m):
        y = yaw(m)
        # integrate continuous yaw changes
        if self.o is not None:
            self.so += wrap(y-self.o)     # unwrap by summing small wrapped steps
        self.o = y

    def on_gt(self, m):
        y = yaw(m)
        #integrate continuous ground truth yaw changes
        if self.g is not None:
            self.sg += wrap(y - self.g)
        self.g = y

    def cmd(self, w):
        #publish angular velocity command
        t = Twist()
        t.angular.z = w
        self.pub.publish(t)

    def step(self):
        now = self.get_clock().now().nanoseconds / 1e9
        #waiting for initial odom and ground truth messages
        if self.state == 'wait' and self.o is not None and self.g is not None:
            self.so0, self.sg0, self.state = self.so, self.sg, 'spin'
        #spinning until odometry reports 2π rad of rotation
        elif self.state == 'spin':
            if abs(self.so - self.so0) >= 2 * math.pi:
                self.cmd(0.0)
                self.t_stop, self.state = now, 'rest'
            else:
                self.cmd(self.w)

        #resting for 2 seconds to allow the robot to come to a stop
        elif self.state == 'rest':
            self.cmd(0.0)
            if now - self.t_stop >= 2.0:    #2 s to come to rest
                do, dg = self.so - self.so0, self.sg - self.sg0
                print(f"odom: {do:.4f} rad ({math.degrees(do):.2f}°) | "
      f"ground_truth: {dg:.4f} rad ({math.degrees(dg):.2f}°) | "
      f"ratio (truth/odom): {dg / do:.4f}")
                self.done = True

def main():
    rclpy.init()
    n = SpinTest()
    while rclpy.ok() and not n.done:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()