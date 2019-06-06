#!/usr/bin/env python
import rospy
import rosbag
from sensor_msgs.msg import Image

enable_rgb = False
enable_depth = False
enable_infra = False

class Enrol(object):

    def __init__(self):

        """Node Parameters"""

        self.cedula = (raw_input("Introduzca la cedula: "))
        self.path = ('/home/innovacion/Bags/' + self.cedula + '.bag')
        self.bag =  rosbag.Bag(self.path, 'w')

        print self.path

        """Subscribers"""
        rospy.Subscriber("/camera/depth/image_rect_raw" , Image , self.callback_depth, queue_size=1)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image,  self.callback_infra, queue_size=1)
        rospy.Subscriber("/camera/color/image_raw", Image,  self.callback_rgb, queue_size=1)

        # self.confir = str(raw_input("Avise si termino: "))

    def callback_rgb(self, data):

        global enable_rgb
        enable_rgb = True
        self.rgb = data

    def callback_depth(self, data):

        global enable_depth
        enable_depth = True
        self.depth = data

    def callback_infra(self, data):

        global enable_infra
        enable_infra = True
        self.infra = data

    def main(self):

        self.bag.write('/camera/depth/image_rect_raw', self.depth)
        self.bag.write('/camera/infra1/image_rect_raw', self.infra)
        self.bag.write('/camera/color/image_raw', self.rgb)
    
if __name__ == '__main__':

    rospy.init_node('EnrolNode',anonymous=True)
    rospy.loginfo('Enrol Node Started')
    e = Enrol()

    while not rospy.is_shutdown():

        if enable_infra is True and enable_depth is True and enable_rgb is True:

            e.main()
        else:
            pass

    rospy.spin()
