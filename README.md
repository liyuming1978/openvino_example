# cvsdk_example

```markdown
1./multicam_mobilenetssd/dependence  build libyuv and install all  --- kernel must less than 4.10 (4.13 xcb wrong)
2. source /opt/intel/computer_vision_sdk/bin/setupvars.sh 
3. mkdir build, cmake ..
4.  ./intel64/Release/multicam_mobilenetssd ../model/MobileNetSSD_deploy.xml ../model/MobileNetSSD_deploy.bin 1xcb ../video/
   or   ./intel64/Release/multicam_mobilenetssd ../model/MobileNetSSD_deploy.xml ../model/MobileNetSSD_deploy.bin 0 ../video/   if you donot want display
