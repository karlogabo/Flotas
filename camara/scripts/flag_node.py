#!/usr/bin/env python
import roslaunch
import rospy
from std_msgs.msg import Float32

enable_flag = False

class Flag(object):

    def __init__(self):

        """ Subscribers """
        rospy.Subscriber("/flag_pub", Float32, self.callback_flag, queue_size = 1)

        """ Node Parameters """
        self.primer = False


    def callback_flag(self, datos):

        global enable_flag
        enable_flag = True
        self.flag = datos.data

    def main(self):

        if self.flag == 1.0:

            if self.primer is False:
                self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
                roslaunch.configure_logging(self.uuid)
                self.launch =  roslaunch.parent.ROSLaunchParent(self.uuid, ["/home/innovacion/ADAS_workspace/src/camara/launch/grabacion.launch"])
                self.launch.start()
                rospy.loginfo("Grabacion launch started")
                self.primer = True
        else:
            
            if self.primer is True:
                self.launch.shutdown()
                self.primer = False
            # print("Grabacion detenida")

if __name__ == '__main__':

    rospy.init_node('Infrarojo',anonymous=True)
    flag = Flag()
    rospy.loginfo('Flag Node started')

    while not rospy.is_shutdown():

        if enable_flag is True:
            flag.main()

    rospy.spin()
