#!/usr/bin/env python
import math
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

enable_euler = False
class Euler(object):
    def __init__(self):

        """Subscribers"""
        rospy.Subscriber("/camera_imu" , Imu, self.callback_imu, queue_size=1)

        """Publishers"""
        self.euler_angles_yaw = rospy.Publisher("yaw" , Float32, queue_size = 1)
        self.yaw_msg = Float32()
        self.euler_angles_pitch = rospy.Publisher("pitch" , Float32, queue_size = 1)
        self.pitch_msg = Float32()
        self.euler_angles_roll = rospy.Publisher("roll" , Float32, queue_size = 1)
        self.roll_msg = Float32()

        """Node parameters"""
        self.yaw = 0
        self.pitch = 0
        self.roll = 0

    def callback_imu(self, data):

        global enable_euler
        enable_euler =  True
        self.x = data.orientation.x
        self.y = data.orientation.y
        self.z = data.orientation.z
        self.w = data.orientation.w

    def to_euler(self, x, y, z, w):

        sinr_cosp = +2.0 * (w * x + y * z)
        cosr_cosp = +1.0 - 2.0 *( x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = +2.0 * (w * y - z * x)
        if (abs(sinp) >= 1):
            pitch = math.copysign(math.pi / 2, sinp) # // use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        siny_cosp = +2.0 * (w * z + x * y)
        cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (yaw) , (pitch) , (roll)

    def main(self):

        self.yaw, self.pitch, self.roll = self.to_euler(self.x, self.y, self.z, self.w)
        self.yaw_msg.data = self.yaw
        self.euler_angles_yaw.publish(self.yaw_msg)
        self.pitch_msg.data = self.pitch
        self.euler_angles_pitch.publish(self.pitch_msg)
        self.roll_msg.data = self.roll
        self.euler_angles_roll.publish(self.roll_msg)

if __name__ == '__main__':

    rospy.init_node('QuaternionToEuler',anonymous=True)
    rospy.loginfo('Euler To Quaternion Node started')
    angle = Euler()

    while not rospy.is_shutdown():

        if enable_euler is True:
            angle.main()

    rospy.spin()
