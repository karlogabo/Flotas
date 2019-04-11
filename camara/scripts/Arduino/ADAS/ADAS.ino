// Sensors' includes
#include "mpu.h"
#include "I2Cdev.h"

#include <SparkFunLSM9DS1.h>
#define LSM9DS1_M  0x1E // Would be 0x1C if SDO_M is LOW
#define LSM9DS1_AG  0x6B // Would be 0x6A if SDO_AG is LOW
#define DECLINATION -7.5 // Declination (degrees) in Bogotá, COL.

// ROS includes
#include <ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
//#include <geometry_msgs/TwistStamped.h>
#define ROS_BAUD 115200

//Functions declartions
void readyCb( const std_msgs::Bool& msg);
void setupComm();
void recordEnable();

// ROS Objects and variables
ros::NodeHandle  nh;
sensor_msgs::Imu camera_imu;
sensor_msgs::Imu bracket_imu;
//geometry_msgs::TwistStamped car_velocity;
std_msgs::Bool record;

// ROS Publishers
ros::Publisher camera_imu_pub( "camera_imu",&camera_imu );
ros::Publisher bracket_imu_pub( "bracket_imu",&bracket_imu );
//ros::Publisher car_velocity_pub( "car_velocity",&car_velocity );
ros::Publisher record_pub("record" ,  &record);

// ROS Subscribers
ros::Subscriber<std_msgs::Bool> acquisitionEnableSub("ready", &readyCb);

//variables
bool systemReady;
int ret;
const byte interruptPin = 2;
const byte systemStateLED = 10;
const byte recordingLED = 11;
LSM9DS1 imu;
long t;

void setup() {
  //Node configuration
  nh.getHardware()->setBaud( ROS_BAUD );
  nh.initNode();
  nh.advertise( camera_imu_pub );
  nh.advertise( bracket_imu_pub );
  //nh.advertise( car_velocity );
  nh.subscribe( acquisitionEnableSub );

  //MPU9255 configuration
  camera_imu.header.frame_id = "/camera";
//  ret = mympu_open( 200 );
  //LSM9DS1 configuration
  bracket_imu.header.frame_id = "/bracket";
  imu.settings.device.commInterface = IMU_MODE_I2C;
  imu.settings.device.mAddress = LSM9DS1_M;
  imu.settings.device.agAddress = LSM9DS1_AG;
  imu.begin();
  //Interrupts
//  cli();
//  //Pin Change interrupt
//  pinMode(interruptPin, INPUT_PULLUP);
//  record.data = digitalRead(interruptPin);
//  attachInterrupt(digitalPinToInterrupt(interruptPin), recordEnable, CHANGE);
  record.data = true;
//  sei();

  pinMode(systemStateLED, OUTPUT);
  pinMode(recordingLED, OUTPUT);

  digitalWrite(systemStateLED,LOW);
  digitalWrite(recordingLED, LOW);
  t = millis();
  mympu_update();
} 

void loop() {
    if( record.data ){
      //Camera IMU
      ret = mympu_update();
      camera_imu.header.stamp = nh.now();
      if( ret == 0 ){
        camera_imu.linear_acceleration.x = mympu.accel[0];//mympu.ypr[2]; // Acceleration
        camera_imu.linear_acceleration.y = mympu.accel[1];//mympu.ypr[1];
        camera_imu.linear_acceleration.z = mympu.accel[2];//mympu.ypr[0];
    
        camera_imu.angular_velocity.x = mympu.gyro[2]; // Gyro
        camera_imu.angular_velocity.y = mympu.gyro[1];
        camera_imu.angular_velocity.z = mympu.gyro[0];
        
        camera_imu.orientation.x = mympu.qx; // Orientation
        camera_imu.orientation.y = mympu.qy;
        camera_imu.orientation.z = mympu.qz;
        camera_imu.orientation.w = mympu.qw;
      }

      //Bracket IMU
      
      bracket_imu.header.stamp = nh.now();
      readBracketIMU();

      //GPS

      //Publish
      camera_imu_pub.publish(&camera_imu);
      bracket_imu_pub.publish(&bracket_imu);
      //car_velocity_pub.publish(&car_velocity);
    }
    nh.spinOnce();

}

void readyCb( const std_msgs::Bool& msg){
    systemReady = msg.data;
    if ( systemReady ){
        digitalWrite(systemStateLED,HIGH);
    }else{
        digitalWrite(systemStateLED,LOW);
        digitalWrite(recordingLED, LOW);
        record.data = false;
    }
}

void recordEnable(){
    record.data = digitalRead(interruptPin);
    record_pub.publish(&record);
}

float roll;
float pitch;
float yaw;
void readBracketIMU(){
  imu.readGyro();
  imu.readAccel();
  imu.readMag();

  roll = atan2(imu.ay, imu.az);
  pitch = atan2(-imu.ax, sqrt(imu.ay * imu.ay + imu.az * imu.az));
  
  if (imu.my == 0)
    yaw = (imu.mx < 0) ? PI : 0;
  else
    yaw = atan2(imu.mx, imu.my);
    
  yaw -= DECLINATION * PI / 180;
  
  if (yaw > PI) yaw-= (2 * PI);
  else if (yaw < -PI) yaw += (2 * PI);

  bracket_imu.angular_velocity.x = imu.calcGyro(imu.gx);
  bracket_imu.angular_velocity.y = imu.calcGyro(imu.gy);
  bracket_imu.angular_velocity.z = imu.calcGyro(imu.gz);

  bracket_imu.linear_acceleration.x = imu.calcAccel(imu.ax);
  bracket_imu.linear_acceleration.y = imu.calcAccel(imu.ay);
  bracket_imu.linear_acceleration.z = imu.calcAccel(imu.az);

  toQuaternion(yaw, pitch, roll);
  
}

void toQuaternion(float yaw, float pitch, float roll) // yaw (Z), pitch (Y), roll (X)
{
    // Abbreviations for the various angular functions
    float cy = cos(yaw * 0.5);
    float sy = sin(yaw * 0.5);
    float cp = cos(pitch * 0.5);
    float sp = sin(pitch * 0.5);
    float cr = cos(roll * 0.5);
    float sr = sin(roll * 0.5);

    bracket_imu.orientation.x = cy * cp * sr - sy * sp * cr;
    bracket_imu.orientation.y = sy * cp * sr + cy * sp * cr;
    bracket_imu.orientation.z = sy * cp * cr - cy * sp * sr;
    bracket_imu.orientation.w = cy * cp * cr + sy * sp * sr;
}
