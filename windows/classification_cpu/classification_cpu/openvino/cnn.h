#ifndef __CNN_H_
#define __CNN_H_
#pragma warning(disable:4251)  //needs to have dll-interface to be used by clients of class 

#include <vector>
#include <queue>
#include <atltypes.h>
#include <ie_plugin_config.hpp>
#include <ie_plugin_ptr.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <inference_engine.hpp>

#include <sys/stat.h>
#include <ext_list.hpp>

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
	
using std::string;
using namespace InferenceEngine::details;
using namespace InferenceEngine;

class CNNClass {
public:
	typedef enum {
		INSERTIMG_NULL=-1,
		INSERTIMG_INSERTED=0,
		INSERTIMG_PROCESSED=1,
	}InsertImgStatus;

	CNNClass();
	bool Load(const string& model_file,const string& weights_file, const string& name);
	~CNNClass();

    inline CSize GetNetSize() { return input_geometry_; }
	void Flush();
	InsertImgStatus InsertImage();

	string err_msg;

private:
	string inputname;
	string outputname;
	string workname;
    CSize input_geometry_;
	bool bLoad;
	//-- must be global, else it will release... !!! note the sequence is important
	InferenceEngine::InferencePlugin plugin;
	InferenceEngine::CNNNetwork network_;
	InferenceEngine::ExecutableNetwork exenet;
	InferRequest::Ptr infer_request_curr_;
	InferRequest::Ptr infer_request_next_;
};


#endif //__CNN_H_
