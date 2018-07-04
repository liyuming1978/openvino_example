#include "landmark.h"
LandMark::LandMark()
{
	bLoad = false;
}

void LandMark::Load(const string& model_file,
	const string& weights_file,int batch)
{
	err_msg = "";
	// --------------------Load network (Generated xml/bin files)-------------------------------------------
	//InferenceEngine::InferencePlugin enginePtr;
	try {
		enginePtr = PluginDispatcher({ "" }).getPluginByDevice("GPU");
	}
	catch (InferenceEngineException e) {
		err_msg = "can not find clDNNPlugin.dll and clDNN64.dll";
		return;
	}
	/** Read network model **/
	InferenceEngine::CNNNetReader netReader;
	try {
		netReader.ReadNetwork(model_file);
		netReader.ReadWeights(weights_file);
	}
	catch (InferenceEngineException e) {
		err_msg = "can not load model";
		return;
	}
	network_ = netReader.getNetwork();
	network_.setBatchSize(batch);
	num_batch_ = (int)network_.getBatchSize();  // dynamically batch (ssd can not, but this can)

	// ---------------------------Set inputs ------------------------------------------------------	
	InferenceEngine::InputsDataMap inputInfo(network_.getInputsInfo());
	auto& inputInfoFirst = inputInfo.begin()->second;
#ifdef INPUT_U8
	inputInfoFirst->setPrecision(Precision::U8); //since mean and scale, here must set FP32
#else
	inputInfoFirst->setPrecision(Precision::FP32); //since mean and scale, here must set FP32
#endif
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
		exenet = enginePtr.LoadNetwork(network_, {});
	}
	catch (InferenceEngineException e) {
		err_msg = "can not find intel GPU, or you need update GPU driver";
		return;
	}
	infer_request_curr_ = exenet.CreateInferRequestPtr();
	Blob::Ptr imageInput = infer_request_curr_->GetBlob(inputname);
	num_channels_ = (int)imageInput->dims()[2];
	if (!(num_channels_ == 3 || num_channels_ == 1))
		throw std::logic_error("Input layer should have 1 or 3 channels");
	input_geometry_ = cv::Size((int)imageInput->dims()[0], (int)imageInput->dims()[1]);
	WrapInputLayer(static_cast<IDtype*>(infer_request_curr_->GetBlob(inputname)->buffer()));
	nbatch_index_ = 0;
	bLoad = true;
}

LandMark::~LandMark()
{
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation */
void LandMark::WrapInputLayer(IDtype* input_data) {
	input_channels.clear();
	int width = input_geometry_.width;
	int height = input_geometry_.height;
	for (int i = 0; i < num_batch_*num_channels_; ++i) {
#ifdef INPUT_U8
		cv::Mat channel(height, width, CV_8UC1, input_data);
#else
		cv::Mat channel(height, width, CV_32FC1, input_data);
#endif
		input_channels.push_back(channel);
		input_data += width * height;
	}
}

cv::Mat LandMark::PreProcess(const cv::Mat& img) {
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_) {
		cv::resize(sample, sample_resized, input_geometry_);
	}
	else
		sample_resized = sample;

#ifdef INPUT_U8
	return sample_resized;
#else
	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	/* Convert the input image to the input image format of the network. */
	cv::scaleAdd(sample_float, 0.007843, mean_, sample_float); //scaleAdd or (add+multiply)? which speed : ans is scaleAdd

	return sample_float;
#endif
}

//note: landmark is (-128,128) not (-1,l), becuase (-1,1) fp16 has poor accuracy
float* LandMark::getLandmark()
{
	infer_request_curr_->Infer();
	nbatch_index_ = 0;
	const Blob::Ptr output_blob = infer_request_curr_->GetBlob(outputname);
	return static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());
}

//-- now, it dosen't work.
void LandMark::setBatch(int batch)
{
	if (batch != num_batch_ && batch>0) {
		num_batch_ = batch;
		network_.setBatchSize(batch);
		WrapInputLayer(static_cast<IDtype*>(infer_request_curr_->GetBlob(inputname)->buffer()));
	}
	nbatch_index_ = 0;
}

bool LandMark::insertImage(const cv::Mat& orgimg)
{
	if (orgimg.cols == 0 || orgimg.rows == 0 || nbatch_index_ >= num_batch_ ||!bLoad) {
		return false;
	}

	cv::Mat img = PreProcess(orgimg);
	cv::split(img, &input_channels[num_channels_*nbatch_index_]);
	nbatch_index_++;
	return true;
}