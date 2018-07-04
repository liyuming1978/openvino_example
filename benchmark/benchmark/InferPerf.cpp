#include "InferPerf.h"
#include <chrono>
#include <ext_list.hpp>

InferPerf::InferPerf()
{
	bLoad = false;
}

InferPerf::~InferPerf()
{
}

void InferPerf::Load(const string& pathname,
	const string& modelname, const string& device, int batch)
{
	err_msg = "";
	// --------------------Load network (Generated xml/bin files)-------------------------------------------
	try {
		plugin_ = PluginDispatcher({ "" }).getPluginByDevice(device);
	}
	catch (InferenceEngineException e) {
		err_msg = e.what();
		return;
	}
	/** Loading default extensions **/
	if (device.find("CPU") != std::string::npos) {
		/**
			* cpu_extensions library is compiled from "extension" folder containing
			* custom MKLDNNPlugin layer implementations. These layers are not supported
			* by mkldnn, but they can be useful for inferring custom topologies.
		**/
		plugin_.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
	}	
	
	// Read network model 
	InferenceEngine::CNNNetReader netReader;
	try {
		netReader.ReadNetwork(pathname+ modelname+".xml");
		netReader.ReadWeights(pathname + modelname + ".bin");
	}
	catch (InferenceEngineException e) {
		err_msg = e.what();
		return;
	}
	network_ = netReader.getNetwork();
	network_.setBatchSize(batch);
	num_batch_ = (int)network_.getBatchSize();  // dynamically batch (ssd can not, but time ok)

 // ---------------------------Set inputs ------------------------------------------------------	
	InferenceEngine::InputsDataMap inputInfo(network_.getInputsInfo());
	auto& inputInfoFirst = inputInfo.begin()->second;
	inputInfoFirst->setPrecision(Precision::U8); //since mean and scale, here must set FP32
	//inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
	inputname = inputInfo.begin()->first;
	// ---------------------------Set outputs ------------------------------------------------------	
	InferenceEngine::OutputsDataMap outputInfo(network_.getOutputsInfo());
	auto& _output = outputInfo.begin()->second;
	_output->setPrecision(Precision::FP32);
	//_output->setLayout(Layout::NCHW);	
	outputname = outputInfo.begin()->first;
	const InferenceEngine::SizeVector outputDims = _output->dims;
	// -------------------------Loading model to the plugin-------------------------------------------------
	try {
		exenet_ = plugin_.LoadNetwork(network_, {});
	}
	catch (InferenceEngineException e) {
		err_msg = e.what();
		return;
	}
	infer_request_curr_ = exenet_.CreateInferRequestPtr();
	Blob::Ptr imageInput = infer_request_curr_->GetBlob(inputname);
	num_channels_ = (int)imageInput->dims()[2];
	if (!(num_channels_ == 3 || num_channels_ == 1))
		throw std::logic_error("Input layer should have 1 or 3 channels");
	nbatch_index_ = 0;
	bLoad = true;
}

string InferPerf::Perf()
{
	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
	typedef std::chrono::duration<float> fsec;

	auto t0 = Time::now();
	for (int i = 0; i < 100; i++) {
		infer_request_curr_->Infer();
		const Blob::Ptr output_blob = infer_request_curr_->GetBlob(outputname);
		float* fdata = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());
		if (fdata[0] > 10000000)  //avoid optimize , get right time
			printf("%f", fdata[0]);
	}
	auto t1 = Time::now();
	fsec fs = t1 - t0;
	ms d = std::chrono::duration_cast<ms>(fs);
	double totalms = d.count();
	char retch[50];
	sprintf (retch,"%.2f",1000 * 100 * num_batch_ / totalms);

	return retch;
}