//Includes
#include "mpu.h"
#include "I2Cdev.h"
#include "Wire.h"
#include <SparkFunLSM9DS1.h>
#define LSM9DS1_M  0x1E // Would be 0x1C if SDO_M is LOW
#define LSM9DS1_AG  0x6B // Would be 0x6A if SDO_AG is LOW
#define DECLINATION -7.5 // Declination (degrees) in Bogotá, COL.

#define    MPU9250_ADDRESS            0x68
#define    MAG_ADDRESS                0x0C
#define    GYRO_FULL_SCALE_250_DPS    0x00  
#define    GYRO_FULL_SCALE_500_DPS    0x08
#define    GYRO_FULL_SCALE_1000_DPS   0x10
#define    GYRO_FULL_SCALE_2000_DPS   0x18
#define    ACC_FULL_SCALE_2_G        0x00  
#define    ACC_FULL_SCALE_4_G        0x08
#define    ACC_FULL_SCALE_8_G        0x10
#define    ACC_FULL_SCALE_16_G       0x18

// ROS includes
#include <ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
//#include <geometry_msgs/TwistStamped.h>
#define ROS_BAUD 115200

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


LSM9DS1 imu;
long t;


void I2Cread(uint8_t Address, uint8_t Register, uint8_t Nbytes, uint8_t* Data)
{
  // Set register address
  Wire.beginTransmission(Address);
  Wire.write(Register);
  Wire.endTransmission();
  
  // Read Nbytes
  Wire.requestFrom(Address, Nbytes); 
  uint8_t index=0;
  while (Wire.available())
    Data[index++]=Wire.read();
}

void I2CwriteByte(uint8_t Address, uint8_t Register, uint8_t Data)
{
  // Set register address
  Wire.beginTransmission(Address);
  Wire.write(Register);
  Wire.write(Data);
  Wire.endTransmission();
}

void setup() {

  nh.getHardware()->setBaud( ROS_BAUD );
  nh.initNode();
  nh.advertise( camera_imu_pub );
  nh.advertise( bracket_imu_pub );
  camera_imu.header.frame_id = "/camera";
  bracket_imu.header.frame_id = "/bracket";

  Wire.begin();
  Serial.begin(115200);
  I2CwriteByte(MPU9250_ADDRESS,29,0x06);
  I2CwriteByte(MPU9250_ADDRESS,26,0x06);
  I2CwriteByte(MPU9250_ADDRESS,27,GYRO_FULL_SCALE_1000_DPS);
  I2CwriteByte(MPU9250_ADDRESS,28,ACC_FULL_SCALE_4_G);
  I2CwriteByte(MPU9250_ADDRESS,0x37,0x02);
  I2CwriteByte(MAG_ADDRESS,0x0A,0x16);

  imu.settings.device.commInterface = IMU_MODE_I2C;
  imu.settings.device.mAddress = LSM9DS1_M;
  imu.settings.device.agAddress = LSM9DS1_AG;
  imu.begin();

}

void loop() {

  uint8_t Buf[14];
  I2Cread(MPU9250_ADDRESS,0x3B,14,Buf);
  
  // Create 16 bits values from 8 bits data
  
  // Accelerometer
  int16_t ax=-(Buf[0]<<8 | Buf[1]);
  int16_t ay=-(Buf[2]<<8 | Buf[3]);
  int16_t az=Buf[4]<<8 | Buf[5];

  // Gyroscope
  int16_t gx=-(Buf[8]<<8 | Buf[9]);
  int16_t gy=-(Buf[10]<<8 | Buf[11]);
  int16_t gz=Buf[12]<<8 | Buf[13];
  
  // Display values
  
  // Accelerometer

  
  camera_imu.linear_acceleration.x = (ax);
  camera_imu.linear_acceleration.y = (ay); 
  camera_imu.linear_acceleration.z = (az); 
  camera_imu.angular_velocity.x = (gx); // Gyro
  camera_imu.angular_velocity.y = (gy);
  camera_imu.angular_velocity.z = (gz);
  
  readBracketIMU();
  camera_imu_pub.publish(&camera_imu);
  bracket_imu_pub.publish(&bracket_imu);
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
