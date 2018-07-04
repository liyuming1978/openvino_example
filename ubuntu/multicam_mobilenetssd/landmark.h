#ifndef __LANDMARK_H_
#define __LANDMARK_H_
#pragma warning(disable:4251)  //needs to have dll-interface to be used by clients of class 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <queue>
#include <ie_plugin_config.hpp>
#include <ie_plugin_ptr.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <inference_engine.hpp>

/**************************************************************************************************
python3  /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_caffe.py
-m MobileNetSSD_deploy.caffemodel --data_type FP16 -b 6 --mean_values [1,1,1] -s 127.5

-b 6 is must , IE only support fixed batch into model(IR)

caffe: mean 127.5 scale 0.007843
but IE: scale 127.5 (1/0.007843)  mean 1 (127.5*0.007843)
if you include mean and scale with IR, the input will be U8 (#define INPUT_U8)
****************************************************************************************************/

#define INPUT_U8
#ifdef INPUT_U8
typedef  unsigned char IDtype;
#else
typedef  float IDtype;
#endif

using std::queue;
using std::string;
using std::vector;
using namespace InferenceEngine::details;
using namespace InferenceEngine;
#define LANDMARK_COUNT 136

class LandMark
{
public:
	void Load(const string& model_file,
		const string& weights_file,int batch=1);
	LandMark();
	~LandMark();
	void setBatch(int batch);
	bool insertImage(const cv::Mat& orgimg);
	float* getLandmark();
	std::string err_msg;

private:
	void WrapInputLayer(IDtype* input_data);
	cv::Mat PreProcess(const cv::Mat& img);
	int num_batch_;
	int nbatch_index_;
	std::vector<cv::Mat> input_channels;
	cv::Size input_geometry_;
	int num_channels_;
	std::string inputname;
	std::string outputname;
	//-- must be global, else it will release... !!! note the sequence is important
	InferenceEngine::InferencePlugin enginePtr;
	InferenceEngine::CNNNetwork network_;
	InferenceEngine::ExecutableNetwork exenet;
	InferRequest::Ptr infer_request_curr_;
	InferRequest::Ptr infer_request_next_;
	bool bLoad;
};

#endif //__LANDMARK_H_

