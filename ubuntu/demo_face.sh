#!/bin/bash
cd ~/work/cvsdk_example2/build
source /opt/intel/computer_vision_sdk/bin/setupvars.sh 
./intel64/Release/multicam_mobilenetssd ../model/facedet6.xml ../model/facedet6.bin 1xcb ../video/ 


