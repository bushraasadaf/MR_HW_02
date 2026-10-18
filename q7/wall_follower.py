import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import csv

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')
        #whenevr a new /scan arrives, call the scan_callback function
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        #publisher for robot velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.D_ref = 0.5 
        self.v= 0.2
        self.L = 0.5 #Lookahead length for heading error 
        # self.kp = 1.0
        self.kp = 3.0
        self.kd=0.05
        self.ki=0.0
        self.prev_error = 0.0  
        self.integral = 0.0 #for the integral
        self.prev_time = None #for dt
        self.t0 = None

        #create csv to log data (time, D, alpha, e, omega)
        # self.csv_file = open('wall_follower_log.csv', 'w', newline='')
        self.csv_file = open(f'q7_e_{self.kd}.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'D', 'alpha', 'e', 'omega']) 

    def scan_callback(self, msg):
        #theta = 45 deg: beam b = ranges[270], beam a = ranges[315]
        #replaced inf with range_max instead of returning, so the robot keeps getting commands
        a = msg.ranges[315] if math.isfinite(msg.ranges[315]) else msg.range_max
        b = msg.ranges[270] if math.isfinite(msg.ranges[270]) else msg.range_max
        theta = math.radians(45)

        alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))
        D = b * math.cos(alpha)
        #e = Dref − D, with Dref = 0.5 m (the robot starts about 0.8 m from the wall
        #and the question says it begins 0.3 m too far so Dref=0.5m)
        e = self.D_ref - D - self.L*math.sin(alpha) #e = pos error + heading error

        t = self.get_clock().now().nanoseconds / 1e9
        if self.t0 is None:
            self.t0 = t

        dt = 0.0 if self.prev_time is None else t - self.prev_time

        if dt > 0:
            derivative = (e - self.prev_error) / dt
            self.integral += e * dt
        else:
            derivative = 0.0

        omega = self.kp * e + self.kd * derivative + self.ki * self.integral
        omega = max(-1.5, min(1.5, omega))

        cmd = Twist()
        cmd.linear.x = self.v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

        self.csv_writer.writerow([t - self.t0, D, alpha, e, omega])
        self.csv_file.flush()

        #update state for the next call 
        self.prev_time = t
        self.prev_error = e

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cmd_pub.publish(Twist())
    node.csv_file.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()