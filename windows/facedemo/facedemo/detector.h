#ifndef __DETECTOR_H_
#define __DETECTOR_H_
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

class Detector {
public:
	typedef enum {
		INSERTIMG_NULL=-1,
		INSERTIMG_INSERTED=0,
		INSERTIMG_PROCESSED=1,
		INSERTIMG_GET=2,
	}InsertImgStatus;

	typedef struct __ImageInfo {
		cv::Size isize;
		int inputid;
		cv::Mat orgimg;
	}ImageInfo;

	typedef struct __resultbox {
		int classid;
		float confidence;
		int left;
		int right;
		int top;
		int bottom;
	}resultbox;

	typedef struct __Result {
		vector<resultbox> boxs;
		cv::Mat orgimg;
		cv::Size imgsize;
		int inputid;
	}DetctorResult;
	
	Detector();
	void Load(const string& model_file,const string& weights_file);
	~Detector();

	inline int GetCurBatch(){return  num_batch_;}
	inline cv::Size GetNetSize(){return input_geometry_;}
	InsertImgStatus InsertImage(const cv::Mat& orgimg, vector<DetctorResult>& objects, int inputid = 0);
	void SwitchMode();
	std::string err_msg;

private:
	void WrapInputLayer(IDtype* input_data);
	cv::Mat PreProcess(const cv::Mat& img);
	void CreateMean();
	void EmptyQueue(queue<ImageInfo>& que);
	std::vector<cv::Mat> input_channels;
	cv::Size input_geometry_;
	queue<ImageInfo> imginfoque_;
	int nbatch_index_;
	int num_channels_;
	int num_batch_;
	cv::Mat mean_;
	std::string inputname;
	std::string outputname;
	int maxProposalCount;
	int objectSize;
	bool bisSync;
	bool bLoad;
	//-- must be global, else it will release... !!! note the sequence is important
	InferenceEngine::InferencePlugin enginePtr;
	InferenceEngine::CNNNetwork network_;
	InferenceEngine::ExecutableNetwork exenet;
	InferRequest::Ptr infer_request_curr_;
	InferRequest::Ptr infer_request_next_;
};


#endif //__DETECTOR_H_