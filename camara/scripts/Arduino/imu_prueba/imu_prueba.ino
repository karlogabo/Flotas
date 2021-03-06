#include <ros.h>
#include <std_msgs/Float32.h>
#include <Adafruit_LSM9DS1.h>

ros::NodeHandle  nh;
Adafruit_LSM9DS1 lsm = Adafruit_LSM9DS1();
//std_msgs::Float32 imu_msg;
std_msgs::Float32 imu_msgy;
std_msgs::Float32 imu_msgz;
std_msgs::Float32 FLAG_msg;
//ros::Publisher accel_x("accel_x", &imu_msg);
ros::Publisher accel_y("accel_y", &imu_msgy);
ros::Publisher accel_z("accel_z", &imu_msgz);
ros::Publisher flag_pub("flag_pub" ,  &FLAG_msg);



void setupSensor(){  
  lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_2G);    //Rango acelerometro
  lsm.setupMag(lsm.LSM9DS1_MAGGAIN_4GAUSS);     //Sensibilidad magnetometro
  lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_245DPS);  //Configruaciones giroscopio
}

void setup() {
  pinMode(13, OUTPUT);
  nh.initNode();  
  //nh.advertise(accel_x);
  nh.advertise(accel_y);
  nh.advertise(accel_z);
  nh.advertise(flag_pub);
  lsm.begin();
  setupSensor();
}

void loop() {
  lsm.read();
  FLAG_msg.data = digitalRead(7);
  sensors_event_t a, m, g, temp;
  lsm.getEvent(&a, &m, &g, &temp);  
  //imu_msg.data = a.acceleration.x;
  imu_msgy.data = a.acceleration.y;
  imu_msgz.data = a.acceleration.z;
  //accel_x.publish( &imu_msg );
  accel_y.publish( &imu_msgy );
  accel_z.publish( &imu_msgz );
  flag_pub.publish( &FLAG_msg ); 
  nh.spinOnce();
  
}
