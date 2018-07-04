#!/bin/bash
source /opt/intel/computer_vision_sdk/bin/setupvars.sh 
python3  /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_caffe.py -m caffe/landmark.caffemodel --data_type FP16 -b 1 --mean_values [1,1,1] -s 127.5	


