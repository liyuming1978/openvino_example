#!/bin/bash
cd ~/work/cvsdk_example2/build
source /opt/intel/computer_vision_sdk/bin/setupvars.sh 
./intel64/Release/multicam_mobilenetssd ../model/MobileNetSSD_deploy.xml ../model/MobileNetSSD_deploy.bin 1xcb ../video/
 


