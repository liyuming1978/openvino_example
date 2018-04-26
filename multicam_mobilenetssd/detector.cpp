#include "detector.hpp"

Detector::Detector(const string& model_file,
	const string& weights_file,bool keep_orgimg) {
	
	pthread_mutex_init(&mutex,NULL); 
	keep_orgimg_ = keep_orgimg;
// --------------------Load network (Generated xml/bin files)-------------------------------------------
	InferenceEngine::InferencePlugin enginePtr = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice("GPU");
/** Read network model **/  //how complex api and var.   yuming.li mark:  I fu you...
	//std::string inputname;
	//std::string outputname;
	InferenceEngine::CNNNetReader netReader;
	netReader.ReadNetwork(model_file);
	netReader.ReadWeights(weights_file);
	network_ = netReader.getNetwork();
	//SetBatch(6); //just set , you can set any value here. or not set , just keep net batch? -- yuming.li mark
	num_batch_ = network_.getBatchSize();  //IE can not support dynamically batch for ssd
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
	maxProposalCount = outputDims[1];	//liyuming mark: if detection > maxProposalCount... 
// -------------------------Loading model to the plugin-------------------------------------------------
	InferenceEngine::ExecutableNetwork exenet = enginePtr.LoadNetwork(network_, {});
	infer_request_ = exenet.CreateInferRequestPtr();	
	Blob::Ptr imageInput = infer_request_->GetBlob(inputname);
	num_channels_ = imageInput->dims()[2];
	if(!(num_channels_ == 3 || num_channels_ == 1))
		throw std::logic_error("Input layer should have 1 or 3 channels");
	input_geometry_ = cv::Size(imageInput->dims()[0], imageInput->dims()[1]);	
	CreateMean();
	
	nbatch_index_ = 0;
	m_start = true;

	pnet_data = static_cast<IDtype*>(imageInput->buffer());
#ifndef UGLY_NO_COPY
	pbatch_element_ = NULL;
#else
	WrapInputLayer(pnet_data);
#endif	
	sem_init(&insert_semt_, 0, 0);
	sem_post(&insert_semt_);
}

Detector::~Detector() {
	pthread_mutex_destroy(&mutex); 
	EmptyQueue(batchque_);
	EmptyQueue(imgsizeque_);
	EmptyQueue(imgque_);	
}

void Detector::EmptyQueue(queue<Detector::ImageSize>& que)
{
	while(!que.empty()){
		que.pop();
	}
}

void Detector::EmptyQueue(queue<cv::Mat>& que)
{
	while(!que.empty()){
		que.pop();
	}
}

void Detector::EmptyQueue(queue<IDtype*>& que)
{
	while(!que.empty()){
#ifndef UGLY_NO_COPY
		delete que.front();
#endif
		que.pop();
	}	
}

//note! do not call it if Detect not finished
void Detector::SetBatch(int batch)
{
	if(batch<1 ||batch==num_batch_)
		return;
	num_batch_=batch;
	//network_.setBatchSize(num_batch_);  //IE can not support dynamically batch for ssd
}

int Detector::TryDetect() {
	int curbatch=0;
	pthread_mutex_lock(&mutex); 
	if(!batchque_.empty())
		//curbatch=batchque_.front().num;
		curbatch = num_batch_;
	pthread_mutex_unlock(&mutex);
	return curbatch;
}

bool Detector::Detect(vector<Detector::Result>& objects) {
	IDtype* pdata;
	int nbatchnum=num_batch_;
	pthread_mutex_lock(&mutex); 
	if(!batchque_.empty()){
		pdata = batchque_.front();
		batchque_.pop();
	}
	else{
		pthread_mutex_unlock(&mutex); 	
		return false;
	}

	for (int i=0;i<nbatchnum;i++) {
		if(!imgsizeque_.empty()){
			objects[i].imgsize = imgsizeque_.front().isize;
			objects[i].inputid = imgsizeque_.front().inputid;
			imgsizeque_.pop();
		}
		else
			objects[i].imgsize = cv::Size(0,0);
		
		if(keep_orgimg_ && !imgque_.empty()){
			objects[i].orgimg = imgque_.front();
			imgque_.pop();
		}
	}
	sem_post(&insert_semt_);
	//SetBatch(nbatchnum);
	pthread_mutex_unlock(&mutex); 
#ifndef UGLY_NO_COPY
	memcpy(pnet_data,pdata,nbatchnum*num_channels_*input_geometry_.height*input_geometry_.width*sizeof(IDtype)); //waist time...
	delete pdata;
#endif

	infer_request_->Infer();
	/* get the result */
	const Blob::Ptr output_blob = infer_request_->GetBlob(outputname);
	const float* result = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());
	for (int k = 0; k < maxProposalCount ; k++) {
		resultbox object;
		int imgid = (int)result[0];
		if (imgid < 0|| result[2] == 0) {  //!!!can not break, naocan  design...
				result+=7;
				continue;
		}		
		int w=objects[imgid].imgsize.width;
		int h=objects[imgid].imgsize.height;		
		object.classid = (int)result[1];
		object.confidence = result[2];
		object.left = (int)(result[3] * w);
		object.top = (int)(result[4] * h);
		object.right = (int)(result[5] * w);
		object.bottom = (int)(result[6] * h);
		if (object.left < 0) object.left = 0;
		if (object.top < 0) object.top = 0;
		if (object.right >= w) object.right = w - 1;
		if (object.bottom >= h) object.bottom = h - 1;
		objects[imgid].boxs.push_back(object);
		result+=7;
	}
	return true;
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation */
void Detector::WrapInputLayer(IDtype* input_data) {
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

cv::Mat Detector::PreProcess(const cv::Mat& img) {
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
	cv::scaleAdd (sample_float, 0.007843, mean_, sample_float); //scaleAdd or (add+multiply)? which speed : ans is scaleAdd
	
	return sample_float;
#endif
}

void Detector::CreateMean() {
#ifdef INPUT_U8
	if (num_channels_ == 3)
		mean_= cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(-127.5*0.007843,-127.5*0.007843,-127.5*0.007843));
	else
		mean_= cv::Mat(input_geometry_, CV_32FC1, cv::Scalar(-127.5*0.007843));	
#endif
}

void Detector::Stop()
{
	m_start=false;
	sem_post(&insert_semt_);
}
/*
	InsertImage will fill a blob until blob full, if full return the blob point
*/
Detector::InsertImgStatus Detector::InsertImage(const cv::Mat& orgimg,int inputid) {
	InsertImgStatus retvalue=INSERTIMG_INSERTED;
	if(orgimg.cols==0 || orgimg.rows==0 ||!m_start)
		return INSERTIMG_NULL;
	
	pthread_mutex_lock(&mutex); 	
	if(nbatch_index_==0){  //new a blob
		pthread_mutex_unlock(&mutex);
		sem_wait(&insert_semt_);
		usleep(1000);  //liyuming mark:  ugly code, this sleep must need to wait the pnet_data will process by first layer
		pthread_mutex_lock(&mutex);
		curdata_batch_ = num_batch_; //fix batch...
#ifndef UGLY_NO_COPY
		if(!pbatch_element_)
			delete pbatch_element_;
		pbatch_element_ = new IDtype[num_batch_*num_channels_*input_geometry_.height*input_geometry_.width];
		WrapInputLayer(pbatch_element_);
#endif		
	}
	ImageSize is;
	is.isize = orgimg.size();
	is.inputid = inputid;
	imgsizeque_.push(is);
	if(keep_orgimg_)
		imgque_.push(orgimg);

	cv::Mat img = PreProcess(orgimg);		
	cv::split(img, &input_channels[num_channels_*nbatch_index_]);

	//if full return pbatch_element_
	if(++nbatch_index_>=num_batch_){
		nbatch_index_=0;
		retvalue = INSERTIMG_FILLONE;
#ifndef UGLY_NO_COPY
		batchque_.push(pbatch_element_);
		pbatch_element_ = NULL;
#else
		batchque_.push((IDtype*)0);
#endif
	}
	pthread_mutex_unlock(&mutex); 	
	return retvalue;
}

