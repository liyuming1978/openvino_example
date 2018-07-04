#pragma once
#include <ie_plugin_config.hpp>
#include <ie_plugin_ptr.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <inference_engine.hpp>

#include <string>
using std::string;
using namespace InferenceEngine::details;
using namespace InferenceEngine;

class InferPerf
{
public:
	InferPerf();
	~InferPerf();
	void Load(const string& pathname,
		const string& modelname, const string& device, int batch);
	string Perf();
	string err_msg;

private:
	int num_batch_;
	int nbatch_index_;
	int num_channels_;
	std::string inputname;
	std::string outputname;
	//-- must be global, else it will release... !!! note the sequence is important
	InferenceEngine::InferencePlugin plugin_;
	InferenceEngine::CNNNetwork network_;
	InferenceEngine::ExecutableNetwork exenet_;
	InferRequest::Ptr infer_request_curr_;
	InferRequest::Ptr infer_request_next_;
	bool bLoad;
};

