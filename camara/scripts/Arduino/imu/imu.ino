#include <Adafruit_LSM9DS1.h>
#include <Adafruit_Sensor.h> 
#include <Wire.h>
#include <ros.h>
#include <geometry_msgs/Pose.h>

Adafruit_LSM9DS1 lsm = Adafruit_LSM9DS1();
//ros::NodeHandle nh;
//geometry_msgs::Pose imu_msg;
//ros::Publisher imu("imu", &imu_msg);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  if (!lsm.begin())
  {
    Serial.println("No se pudo inicializar el sensor");
    while (1);
  }
  //Serial.println("Sensor inicializado");
  //nh.initNode();
  //nh.advertise(imu);
  lsm.begin();
  setupSensor();

}

void setupSensor(){
  
  lsm.setupAccel(lsm.LSM9DS1_ACCELRANGE_2G);    //Rango acelerometro
  lsm.setupMag(lsm.LSM9DS1_MAGGAIN_4GAUSS);     //Sensibilidad magnetometro
  lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_245DPS);  //Configruaciones giroscopio
}

void loop() {

  lsm.read();
  sensors_event_t a, m, g, temp;
  lsm.getEvent(&a, &m, &g, &temp);

  //imu_msg.position.x = a.acceleration.x;
  //imu_msg.position.y = a.acceleration.y;
  //imu_msg.position.z = a.acceleration.z;
  //imu.publish(&imu_msg);

  //Serial.print("Accel X: "); 
  Serial.print(a.acceleration.x); Serial.print(",");
  //Serial.print("\tY: "); 
  Serial.print(a.acceleration.y); Serial.print(",");
  //Serial.print("\tZ: "); 
  Serial.print(a.acceleration.z); 
  //Serial.print("Mag X: "); Serial.print(m.magnetic.x);   Serial.print(" gauss");
  //Serial.print("\tY: "); Serial.print(m.magnetic.y);     Serial.print(" gauss");
  //Serial.print("\tZ: "); Serial.print(m.magnetic.z);     Serial.println(" gauss");
  //Serial.print("Gyro X: "); Serial.print(g.gyro.x);   Serial.print(" dps");
  //Serial.print("\tY: "); Serial.print(g.gyro.y);      Serial.print(" dps");
  //Serial.print("\tZ: "); Serial.print(g.gyro.z);      Serial.println(" dps");
  //Serial.print("Temp: "); Serial.print(temp.temperature);   Serial.print("°C");
  Serial.println();
  delay(200);  

}
