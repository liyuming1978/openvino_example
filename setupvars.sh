INSTALLDIR=/opt/intel/computer_vision_sdk_2018.0.226
export INTEL_CVSDK_DIR=$INSTALLDIR
if [ -e $INSTALLDIR/openvx ]; then
   export LD_LIBRARY_PATH=$INSTALLDIR/openvx/lib:$LD_LIBRARY_PATH
fi

if [ -e $INSTALLDIR/deployment_tools/model_optimizer ]; then
   export LD_LIBRARY_PATH=$INSTALLDIR/deployment_tools/model_optimizer/model_optimizer_caffe/bin:$LD_LIBRARY_PATH
   export ModelOptimizer_ROOT_DIR=$INSTALLDIR/deployment_tools/model_optimizer/model_optimizer_caffe
fi

export InferenceEngine_DIR=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/share

if [[ -f /etc/centos-release ]]; then
    IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/centos_7.3/intel64
elif [[ -f /etc/lsb-release ]]; then
    UBUNTU_VERSION=$(lsb_release -r -s)
    if [[ $UBUNTU_VERSION = "16.04" ]]; then
         IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64
    elif [[ $UBUNTU_VERSION = "14.04" ]]; then
         IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/ubuntu_14.04/intel64
    elif cat /etc/lsb-release | grep -q "Yocto" ; then
         IE_PLUGINS_PATH=$INTEL_CVSDK_DIR/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64
    fi
fi

if [ -e $INSTALLDIR/deployment_tools/inference_engine ]; then
   export LD_LIBRARY_PATH=/opt/intel/opencl:$INSTALLDIR/deployment_tools/inference_engine/external/cldnn/lib:$INSTALLDIR/deployment_tools/inference_engine/external/mkltiny_lnx/lib:$IE_PLUGINS_PATH:$LD_LIBRARY_PATH
fi

if [ -e $INSTALLDIR/opencv ]; then
   export OpenCV_DIR=$INSTALLDIR/opencv/share/OpenCV
   export LD_LIBRARY_PATH=$INSTALLDIR/opencv/lib:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=$INSTALLDIR/opencv/share/OpenCV/3rdparty/lib:$LD_LIBRARY_PATH
fi

export PATH="$INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$PATH"
export PYTHONPATH="$INTEL_CVSDK_DIR/deployment_tools/model_optimizer:$PYTHONPATH"