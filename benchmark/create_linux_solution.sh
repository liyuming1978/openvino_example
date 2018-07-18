#!/bin/bash

ROOT_DIR="$( cd "$( dirname "$0"  )" && pwd  )"
export SOLUTION_DIR64=$ROOT_DIR/build
export InferenceEngine_DIR=/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/share/
export OpenCV_DIR=/opt/intel/computer_vision_sdk/opencv
export InferenceEngine_LibDIR=/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/

cd $ROOT_DIR && cmake -E make_directory $SOLUTION_DIR64 && cd $SOLUTION_DIR64 && cmake .. -DInferenceEngine_LibDIR=${InferenceEngine_LibDIR}
make -j

