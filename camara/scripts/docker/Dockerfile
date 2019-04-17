FROM ros:kinetic-robot

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y      

RUN apt install -y software-properties-common && \
apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE && \
    add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u && \
    rm -f /etc/apt/sources.list.d/realsense-public.list && \
    apt-get update && \
    apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg 

#CMD ["realsense-viewer"]

    
