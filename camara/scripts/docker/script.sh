#! /bin/bash
xhost local:root
docker run -it \
  --net=host \
  --privileged \
  --rm \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --env=unix$DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  --privileged -v /dev/usb/hiddev0 \
  -v /dev/ttyACM0:/dev/ttyACM0 \
  -v /home/innovacion/ADAS_workspace/src/realsense2_camera/:/ADAS_ws/src/realsense2_camera \
  -v /home/innovacion/ADAS_workspace/src/ddynamic_reconfigure/:/ADAS_ws/src/ddynamic_reconfigure \
  -v /home/innovacion/ADAS_workspace/src/camara/:/ADAS_ws/src/camara \
  -v /home/innovacion/Bags/:/root/Bags/ \
  karlogabo/flotas:latest
    #sh -c "chmod +x /ADAS_ws/devel/setup.bash && /ADAS_ws/devel/setup.bash && realsense_viewer" bin/bash

# docker run -it --net=host --env="DISPLAY" --privileged -v /dev/usb/hiddev0 karlogabo/flotas:latest
#docker run -it --net=host --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --env=unix$DISPLAY --privileged -v /dev/usb/hiddev0 karlogabo/flotas:latest
# sudo docker commit 210f4ed5ed53 karlogabo/flotas:latest
#sudo docker push karlogabo/flotas:latest #PUSHHHHH


#git commit -m "number of the commit"
#git push -u origin master
