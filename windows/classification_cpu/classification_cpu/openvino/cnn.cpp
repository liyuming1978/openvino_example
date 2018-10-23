#include "cnn.h"
#include <chrono>

#pragma comment(lib, "inference_engine.lib")
#pragma comment(lib, "libiomp5md.lib")
#pragma comment(lib, "cpu_extension.lib")

CNNClass::CNNClass() { 
	bLoad = false; 
	//channels = 3;  //AV_PIX_FMT_YUV420P
	infer_request_curr_ = NULL;
	infer_request_next_ = NULL;
}

bool CNNClass::Load(const string& model_file, const string& weights_file,const string& name)
{
    if (bLoad)
        return true;
    err_msg = "";
	workname = name;
    // --------------------Load network (Generated xml/bin
    // files)-------------------------------------------
    // InferenceEngine::InferencePlugin enginePtr;
    try
    {
		plugin = PluginDispatcher({""}).getPluginByDevice("CPU");
		plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }
    catch (InferenceEngineException e)
    {
        err_msg = "can not find clDNNPlugin.dll and clDNN64.dll";
        return false;
    }
    /** Read network model **/
    InferenceEngine::CNNNetReader netReader;
    try
    {
        netReader.ReadNetwork(model_file);
        netReader.ReadWeights(weights_file);
    }
    catch (InferenceEngineException e)
    {
        err_msg = "can not load model";
        return false;
    }
    network_ = netReader.getNetwork();
    // num_batch_ = (int)network_.getBatchSize();  //IE can not support dynamically batch for ssd
    // ---------------------------Set inputs ------------------------------------------------------
    InferenceEngine::InputsDataMap inputInfo(network_.getInputsInfo());
    auto& inputInfoFirst = inputInfo.begin()->second;
#ifdef INPUT_U8
    inputInfoFirst->setPrecision(Precision::U8);  // mean and scale move to IE
#else
    inputInfoFirst->setPrecision(Precision::FP32);  // since mean and scale, here must set FP32
#endif
    inputInfoFirst->getInputData()->setLayout(Layout::NCHW);  // default is NCHW
    inputname = inputInfo.begin()->first;
    // ---------------------------Set outputs
    // ------------------------------------------------------
    InferenceEngine::OutputsDataMap outputInfo(network_.getOutputsInfo());
    auto& _output = outputInfo.begin()->second;
    _output->setPrecision(Precision::FP32);
    _output->setLayout(Layout::NCHW);
    outputname = outputInfo.begin()->first;

    // -------------------------Loading model to the
    // plugin-------------------------------------------------
    // InferenceEngine::ExecutableNetwork exenet;
    try
    {
        exenet = plugin.LoadNetwork(network_, {});
    }
    catch (InferenceEngineException e)
    {
        err_msg = e.what();
        return false;
    }
    infer_request_curr_ = exenet.CreateInferRequestPtr();
    infer_request_next_ = exenet.CreateInferRequestPtr();
    Blob::Ptr imageInput = infer_request_curr_->GetBlob(inputname);
    input_geometry_ = CSize((int)imageInput->dims()[0], (int)imageInput->dims()[1]);

    bLoad = true;
	return true;
}

CNNClass::~CNNClass() {
}

void CNNClass::Flush()
{
	if(infer_request_curr_)
		infer_request_curr_->Wait(IInferRequest::WaitMode::RESULT_READY);
	if(infer_request_next_)
		infer_request_next_->Wait(IInferRequest::WaitMode::RESULT_READY);
}

double totalms = 0.0;
int totalcount = 0;
//#define SPLITSCREEN
#define SYNC_MODE
CNNClass::InsertImgStatus CNNClass::InsertImage()  //you need pass img buffer here
{
	InsertImgStatus retvalue = INSERTIMG_INSERTED;
	if ( !bLoad )
		return INSERTIMG_NULL;

	IDtype* pdata = static_cast<IDtype*>(infer_request_curr_->GetBlob(inputname)->buffer());
	//fill pdata by your self

	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
	typedef std::chrono::duration<float> fsec;

	auto t0 = Time::now();
	// if get return INSERTIMG_GET
#ifdef SYNC_MODE
	//infer_request_curr_->StartAsync();
#else
	//infer_request_next_->StartAsync();
#endif
	//if (InferenceEngine::OK == infer_request_curr_->Wait(IInferRequest::WaitMode::RESULT_READY))
	{
		infer_request_curr_->Infer();
		auto t1 = Time::now();
		fsec fs = t1 - t0;
		ms d = std::chrono::duration_cast<ms>(fs);
		totalms += d.count();
		if (++totalcount == 100)
		{
			//printf("%s fps=%.2lf\n", workname.c_str(),1000*100/totalms);
			totalms = 0;
			totalcount = 0;
		}
		retvalue = INSERTIMG_PROCESSED;
	}
#ifndef SYNC_MODE  //No define 
	infer_request_curr_.swap(infer_request_next_);
#endif

	return retvalue;
}
