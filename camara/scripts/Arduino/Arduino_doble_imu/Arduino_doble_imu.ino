//Include
#include "MPU9250.h"
#include "Wire.h"
#include <SparkFunLSM9DS1.h>
#define LSM9DS1_M  0x1E // Would be 0x1C if SDO_M is LOW
#define LSM9DS1_AG  0x6B // Would be 0x6A if SDO_AG is LOW
#define DECLINATION -7.5 // Declination (degrees) in Bogotá, COL.
#include <NMEAGPS.h>

#include <GPSport.h>
static gps_fix  fix;

// ROS includes
#include <ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Imu.h>

#define ROS_BAUD 115200

// ROS Objects and variables
ros::NodeHandle  nh;
sensor_msgs::Imu camera_imu;
sensor_msgs::Imu bracket_imu;
std_msgs::Float32 vel_msg;
std_msgs::Float32 FLAG_msg;
std_msgs::Bool record;


// ROS Publishers
ros::Publisher camera_imu_pub( "camera_imu",&camera_imu );
ros::Publisher bracket_imu_pub( "bracket_imu",&bracket_imu );
//ros::Publisher car_velocity_pub( "car_velocity",&car_velocity );
ros::Publisher record_pub("record" ,  &record);
ros::Publisher vel_msg_pub("vel", &vel_msg);
ros::Publisher flag_pub("flag_pub" ,  &FLAG_msg);

// Node parameters
LSM9DS1 imu;
MPU9250 mpu;
static NMEAGPS  gps;
long t;
float a = 0.000;
String inputString = "";         // a String to hold incoming data
String comparacion = "";
String gp = "$GNRMC";
bool stringComplete = false;

#if !defined( GPS_FIX_SPEED )
  #error You must uncomment GPS_FIX_SPEED in GPSfix_cfg.h!
#endif


static float work( const gps_fix & fix );
static float work( const gps_fix & fix )
{
  if (fix.valid.location) {
    return fix.speed_kph();
  } else {   
  }  
} 

void setup()
{
  
  nh.getHardware()->setBaud( ROS_BAUD );
  nh.initNode();
  nh.advertise( camera_imu_pub );
  nh.advertise( bracket_imu_pub );
  nh.advertise( vel_msg_pub );
  nh.advertise(flag_pub);
  camera_imu.header.frame_id = "/camera";
  bracket_imu.header.frame_id = "/bracket";

  Wire.begin();
//  gpsPort.begin( 38400 );
  Serial1.begin(38400);
  delay(1000);
  mpu.setup();
  imu.settings.device.commInterface = IMU_MODE_I2C;
  imu.settings.device.mAddress = LSM9DS1_M;
  imu.settings.device.agAddress = LSM9DS1_AG;
  imu.begin();
  
}

void loop(){

  FLAG_msg.data = digitalRead(52);

  if (stringComplete) {        
    comparacion = inputString.substring(0,6);
    if (comparacion ==  "$GNRMC"){
      float vel = 0.000;
      vel = inputString.substring(46,51).toFloat();
      vel =  vel * 1.852;
      vel_msg.data = vel; 
    }// clear the string:
    inputString = "";
    stringComplete = false;
  }
  
  while (Serial1.available()) {
    // get the new byte:
    char inChar = (char)Serial1.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
   }

  
   
  mpu.update();
  camera_imu.linear_acceleration.x = (mpu.getAcc(0));
  camera_imu.linear_acceleration.y = (mpu.getAcc(1)); 
  camera_imu.linear_acceleration.z = (mpu.getAcc(2)); 
  camera_imu.angular_velocity.x = (mpu.getGyro(0)); // Gyro
  camera_imu.angular_velocity.y = (mpu.getGyro(1));
  camera_imu.angular_velocity.z = (mpu.getGyro(2));
  camera_imu.orientation.x = mpu.getQuaternion(0); // Orientation
  camera_imu.orientation.y = mpu.getQuaternion(1);
  camera_imu.orientation.z = mpu.getQuaternion(2);
  camera_imu.orientation.w = mpu.getQuaternion(3);
  readBracketIMU();
 
 
  vel_msg_pub.publish(&vel_msg); 
  camera_imu_pub.publish(&camera_imu);
  bracket_imu_pub.publish(&bracket_imu);
  flag_pub.publish( &FLAG_msg );
   
  nh.spinOnce();

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
