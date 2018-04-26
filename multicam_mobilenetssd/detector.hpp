//note: sudo init 3 if you do not want display
//sudo apt-get install compizconfig-settings-manager and do as http://blog.csdn.net/jiankunking/article/details/69467757
//??? low graphic mode in ubuntu unity plugin
//note: speed will very low if screen is lock or dark, so set dark to never and no dim
#ifndef __DETECTOR_HPP_
#define __DETECTOR_HPP_
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV
#include <vector>
#include <queue>
#include <ie_plugin_config.hpp>
#include <ie_plugin_ptr.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <inference_engine.hpp>
#include <unistd.h> 
#include <semaphore.h>

/**************************************************************************************************
python3  /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_caffe.py 
			-m MobileNetSSD_deploy.caffemodel --data_type FP16 -b 6 --mean_values [1,1,1] -s 127.5	
			
-b 6 is must , IE only support fixed batch into model(IR)
			
caffe: mean 127.5 scale 0.007843
but IE: scale 127.5 (1/0.007843)  mean 1 (127.5*0.007843)
if you include mean and scale with IR, the input will be U8 (#define INPUT_U8)

for UGLY_NO_COPY:
    for U8 the mem bandwidth is ok, but for FP32, you'd better use UGLY_NO_COPY
****************************************************************************************************/

#define UGLY_NO_COPY
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
		INSERTIMG_NULL=-2,
		INSERTIMG_BLOCK=-1,
		INSERTIMG_INSERTED=0,
		INSERTIMG_FILLONE=1,
	}InsertImgStatus;

	typedef struct __resultbox {
		float classid;
		float confidence;
		float left;
		float right;
		float top;
		float bottom;
	}resultbox;
	
	typedef struct __Result {
		vector<resultbox> boxs;
		cv::Mat orgimg;
		cv::Size imgsize;
		int inputid;		
	}Result;
	
	typedef struct __ImageSize {
		cv::Size isize;
		int inputid;
	}ImageSize;	
	
	typedef struct __BatchData {
		IDtype* data;
		int num;
	}BatchData;		
	
	Detector(const string& model_file,
		const string& weights_file,bool keep_orgimg);
	~Detector();

	bool Detect(vector<Result>& objects);
	inline int GetCurBatch(){return  num_batch_;}
	inline cv::Size GetNetSize(){return input_geometry_;}
	InsertImgStatus InsertImage(const cv::Mat& orgimg,int inputid);
	int TryDetect();
	void Stop();

private:
	void WrapInputLayer(IDtype* input_data);
	void SetBatch(int batch);	
	cv::Mat PreProcess(const cv::Mat& img);
	void CreateMean();
	void EmptyQueue(queue<ImageSize>& que);
	void EmptyQueue(queue<cv::Mat>& que);
	void EmptyQueue(queue<IDtype*>& que);
	std::vector<cv::Mat> input_channels;
	cv::Size input_geometry_;
	queue<ImageSize> imgsizeque_;
	queue<cv::Mat> imgque_;
	queue<IDtype*> batchque_;	
	pthread_mutex_t mutex; 
	int nbatch_index_;
	int num_channels_;
	//int min_batch_;
	int num_batch_;
	cv::Mat mean_;
	bool keep_orgimg_;
	//int max_imgqueue_;
	int curdata_batch_;
#ifndef UGLY_NO_COPY
	IDtype* pbatch_element_;
#endif
	IDtype* pnet_data;
	bool m_start;
	std::string inputname;
	std::string outputname;
	CNNNetwork network_;
	InferRequest::Ptr infer_request_;
	int maxProposalCount;
	sem_t insert_semt_;
};
#endif //__DETECTOR_HPP_