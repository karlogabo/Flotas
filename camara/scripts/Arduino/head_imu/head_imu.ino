//Includes
#include <WiFi.h>
#include <ros.h>
#include <sensor_msgs/Imu.h>
#include <MPU9250_asukiaaa.h>
#ifdef _ESP32_HAL_I2C_H_
#define SDA_PIN 21
#define SCL_PIN 22
#define DECLINATION -7.5
#endif

//Definitions
MPU9250 mySensor;
const char* ssid     = "Innovacion_MBPO";
const char* password = "1nn0v4c10nMBPO";
IPAddress server(192,168,56,20);
const uint16_t serverPort = 11411;
float roll;
float pitch;
float yaw; 

//ROS Definitions
ros::NodeHandle nh;
sensor_msgs::Imu head_imu;
ros::Publisher head_imu_pub( "head_imu",&head_imu );

void setup() {
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  nh.getHardware()-> setConnection(server, serverPort);
  nh.initNode();
  nh.advertise(head_imu_pub);
  
  //MPU sensor config
  #ifdef _ESP32_HAL_I2C_H_
  // for esp32
  Wire.begin(SDA_PIN, SCL_PIN); //sda, scl
  delay(2000);
  Wire.begin();
  #endif

  mySensor.setWire(&Wire); 
  mySensor.beginAccel();
  mySensor.beginMag();
  mySensor.beginGyro();

  head_imu.header.frame_id = "/head";

}

void loop() {

  headIMU();

  if (nh.connected()) {
    head_imu_pub.publish(&head_imu);  
  } else {
    
  }
  nh.spinOnce();

}

void headIMU(){ 
  head_imu.header.stamp = nh.now();
  mySensor.accelUpdate();
  mySensor.magUpdate();
  mySensor.gyroUpdate();

  roll = atan2(mySensor.accelY(), mySensor.accelZ());
  pitch = atan2(-mySensor.accelX(), sqrt(mySensor.accelY() * mySensor.accelY() + mySensor.accelZ() * mySensor.accelZ()));
  
  if (mySensor.magY() == 0)
    yaw = (mySensor.magX() < 0) ? PI : 0;
  else
    yaw = atan2(mySensor.magX(), mySensor.magY());
    
  yaw -= DECLINATION * PI / 180;
  
  if (yaw > PI) yaw-= (2 * PI);
  else if (yaw < -PI) yaw += (2 * PI);

  head_imu.linear_acceleration.x = mySensor.accelX();
  head_imu.linear_acceleration.y = mySensor.accelY();
  head_imu.linear_acceleration.z = mySensor.accelZ();
  head_imu.angular_velocity.x = mySensor.gyroX();
  head_imu.angular_velocity.y = mySensor.gyroY();
  head_imu.angular_velocity.z = mySensor.gyroZ();
  
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
    
    head_imu.orientation.x = cy * cp * sr - sy * sp * cr;
    head_imu.orientation.y = sy * cp * sr + cy * sp * cr;
    head_imu.orientation.z = sy * cp * cr - cy * sp * sr;
    head_imu.orientation.w = cy * cp * cr + sy * sp * sr;

    
}
