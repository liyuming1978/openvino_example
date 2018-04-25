#include "detector.hpp"
#include "videodec.hpp"
//#include "SDL2Show.hpp"
#include "XCBShow.hpp"
#include <stdio.h>  
#include <pthread.h>  
#include <unistd.h>  
#include <semaphore.h>
#include <chrono>

//--------------------------------------------------------------------
#define INPUTNUM	6
#define RUNWITH_SHOW  //liyuming mark:  delete it if you run without show
const int FIXED_BATCH=INPUTNUM;  //you' better to set equal to INPUTNUM
const int ginput_width=640;
const int ginput_height=480;
bool grunning=false;
Detector* gpdetector;
sem_t g_semt;
int total_frame;
int each_frame[INPUTNUM];
float total_fps;
float each_fps[INPUTNUM];
sem_t g_semtshow;
queue<vector<Detector::Result> > gresultque;
pthread_mutex_t mutexshow ; 
pthread_mutex_t mutexcamimg; 
cv::Mat glastcam;
int isshow=0;
bool bhardware_dec;
enum DISPLAY_TYPE {
	OCVIMGSHOW=0,
	SDLSHOW,
	XCBSHOW,
}gdisplaytype;
static string  CLASSES[] = {"background",
           "aeroplane", "bicycle", "bird", "boat",
           "!bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "!pottedplant",
						"sheep", "sofa", "!train", "tvmonitor"};

void safesleep(int nms)  //I need accurate sleep, so use select,but caffe:timer more accurate
{
#if 0
	struct timeval delay;
	delay.tv_sec = 0;
	delay.tv_usec = nms * 1000; // n ms
	select(0, NULL, NULL, NULL, &delay);
#else
	if(nms<=0)
		return;
	usleep(nms*1000);
#endif
}

void safewakeup()
{
	/*pthread_mutex_lock(&mutex_condition);  
	pthread_cond_signal(&condition); 
	pthread_mutex_unlock(&mutex_condition);	*/
	sem_post(&g_semt);
}

void safewait()
{
	/*pthread_mutex_lock(&mutex_condition);  
	pthread_cond_wait(&condition,&mutex_condition); 
	pthread_mutex_unlock(&mutex_condition);	*/
	sem_wait(&g_semt);
}

void bindcore(pthread_t thrid,int core)
{
	cpu_set_t cpu_info;  
	CPU_ZERO(&cpu_info);  
	CPU_SET(core, &cpu_info);  
	pthread_setaffinity_np(thrid, sizeof(cpu_set_t), &cpu_info);		
}

void *thr_video(void *arg)
{
	//caffe::Timer wait_timer;
	char* videopath = (char*)arg;
	string videoname=videopath;
	videoname+="/";
	VideoDec vdec[INPUTNUM];
	
	for(int i=0;i<INPUTNUM;i++) {
		vdec[i].InitLoad(videoname+std::to_string(i%6)+".mp4",i,bhardware_dec);
	}
	//dec each with 28fps
	int changebatch=0;
	int curbatch=INPUTNUM;
	//wait_timer.Start();
	while (grunning){
		/*changebatch++;
		if(changebatch>1000)
			changebatch=0;
		if((changebatch%2)==0)
			curbatch=INPUTNUM-1;
		else
			curbatch=INPUTNUM;*/
		
		for(int i=0;i<curbatch;i++) {
			//note , here I do each one by one, so avoid heavy IO, and do as quick as possible, no wait
			//cv::Mat frame = vdec[i].GetFrame(gpdetector->GetNetSize().width,gpdetector->GetNetSize().height);
			cv::Mat frame = vdec[i].GetFrame(ginput_width,ginput_height);
			if(gpdetector->InsertImage(frame,i)==Detector::INSERTIMG_FILLONE){
				safewakeup(); 
			}			
		}
		//wait_timer.Stop();
		//double timeUsed = wait_timer.MilliSeconds();
		//std::cout << timeUsed << "\n";
		//wait_timer.Start();
		//safesleep(30-timeUsed);
	}
	return NULL;
}

//http://www.supmen.com/v2p6718wqx.html
//https://stackoverflow.com/questions/25619309/how-do-i-enable-the-uvc-quirk-fix-bandwidth-quirk-in-linux-uvc-driver
//sudo rmmod uvcvideo &  sudo modprobe uvcvideo quirks=640
#if 1 //not faster than in one-thread, but one-thread will very slow, if camer output low
static int gavaiableid=0;
static int gallcount=0;
void *thr_camera_each(void *arg)
{
	int camid=*((int*)arg);
	bool iscam=true;
	
	cv::VideoCapture cap(camid); //cv::CAP_MODE_RGB
	if (!cap.isOpened()) {
		std::cout<< "can not open camera" << camid << " just fake it.\n";
		cap.release(); 
		iscam = false;
	}
	else{
		cap.set(CV_CAP_PROP_FRAME_WIDTH, ginput_width);  
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, ginput_height); 	
		cap.set(CV_CAP_PROP_FPS , 60);
		//if(gpdetector->GetRGBColor()==cv::CAP_MODE_RGB)
		//	cap.set(cv::CAP_PROP_MODE, cv::CAP_MODE_RGB);
		gavaiableid = camid;
		gallcount++;
	}
	
	while (grunning){
		if(iscam){
			cv::Mat frame;
			cap >> frame;
			if(gavaiableid == camid && gallcount!=INPUTNUM){
				pthread_mutex_lock(&mutexcamimg); 	
				glastcam = frame.clone();
				pthread_mutex_unlock(&mutexcamimg); 
			}
			if(gpdetector->InsertImage(frame,camid)==Detector::INSERTIMG_FILLONE){
				safewakeup(); 
			}
		}
		else {
			cv::Mat fakeimg;
			pthread_mutex_lock(&mutexcamimg); 	
			fakeimg = glastcam.clone();
			pthread_mutex_unlock(&mutexcamimg); 	
			if(gpdetector->InsertImage(fakeimg,camid)==Detector::INSERTIMG_FILLONE){
				safewakeup(); 
			}
			safesleep(20);
		}
	}
	if(iscam)
		cap.release(); 
	return NULL;
}

void *thr_camera(void *arg) {
	pthread_t ncamtid[INPUTNUM];
	int camid[INPUTNUM];
	int i;
	cv::Mat lastcamini(ginput_height, ginput_width, CV_8UC3);
	glastcam = lastcamini;
	pthread_mutex_init(&mutexcamimg,NULL); 	
	for(i=0;i<INPUTNUM;i++){
		camid[i]=i;
		pthread_create(&ncamtid[i], NULL, thr_camera_each, (void*)&camid[i]);	
		//bindcore(ncamtid[i],7);
	}
	for(i=0;i<INPUTNUM;i++){
		pthread_join(ncamtid[i],NULL);
	}
	pthread_mutex_destroy(&mutexcamimg); 
	return NULL;
}
#else
void *thr_camera(void *arg) {
	int i;
	cv::VideoCapture cap[INPUTNUM];
	for(i=0;i<INPUTNUM;i++){
		cap[i].open(i);
		if (!cap[i].isOpened()) {
			std::cout<< "can not open camera" << i << " just fake it.\n";
			cap[i].release(); 
		}	
		else {
			//cap[i].set(cv::CAP_PROP_FRAME_WIDTH, 320);  
			//cap[i].set(cv::CAP_PROP_FRAME_HEIGHT, 240); 			
			cap[i].set(CV_CAP_PROP_FPS , 60);		
		}
	}
	
	cv::Mat lastframe;
	while (grunning){
		for(i=0;i<INPUTNUM;i++){
			if (cap[i].isOpened()) {
				cv::Mat frame;  //this is a local var, = new a image
				cap[i].read(frame);
				lastframe = frame;
				if(gpdetector->InsertImage(frame,i)==Detector::INSERTIMG_FILLONE){
					safewakeup(); 
				}				
			}
			else{
				if(gpdetector->InsertImage(lastframe.clone(),i)==Detector::INSERTIMG_FILLONE){
					safewakeup(); 
				}								
			}
		}
	}
	
	//release
	for(i=0;i<INPUTNUM;i++){
		if (cap[i].isOpened())
			cap[i].release(); 
	}	
	return NULL;
}
#endif

void *thr_detector(void *arg)
{
	int fc=0;
	int curbatch;
	while (grunning){
		safewait();
		if(!grunning) break;
		if((curbatch=gpdetector->TryDetect())<=0){
			std::cout << "no data to detect \n" ;
			continue;
		}		
		vector<Detector::Result> objects(curbatch);
		gpdetector->Detect(objects);
		total_frame+=curbatch;	
		for(int k=0;k<objects.size();k++){
			each_frame[objects[k].inputid]+=1;
		}			
#ifdef RUNWITH_SHOW
		if(isshow){
			pthread_mutex_lock(&mutexshow); 	
			gresultque.push(objects);
			pthread_mutex_unlock(&mutexshow); 	
			sem_post(&g_semtshow);
		}
#endif
	}
	return NULL;
}

void *thr_show(void *arg)
{
	bool hasmove=false;
	int fpshaswrite[INPUTNUM];
	cv::Mat onescreen(ginput_height*2, ginput_width*3, CV_8UC3);
	cv::Mat eachscreen[INPUTNUM];
	for(int i=0;i<INPUTNUM;i++)
		eachscreen[i]=onescreen(cv::Rect(ginput_width*(i%3),(i>2)?ginput_height:0,ginput_width,ginput_height));
	
	while(grunning){
		sem_wait(&g_semtshow);
		vector<Detector::Result> objects;
		if(!gresultque.empty()){
			pthread_mutex_lock(&mutexshow); 	
			objects = gresultque.front();	
			gresultque.pop();			
			pthread_mutex_unlock(&mutexshow); 	
			memset(fpshaswrite,0,sizeof(fpshaswrite));
			for(int k=0;k<objects.size();k++){
				for(int i=0;i<objects[k].boxs.size();i++){
					if(objects[k].boxs[i].confidence>0.36 && CLASSES[(int)(objects[k].boxs[i].classid)][0]!='!'){
						cv::rectangle(objects[k].orgimg,cvPoint(objects[k].boxs[i].left,objects[k].boxs[i].top),cvPoint(objects[k].boxs[i].right,objects[k].boxs[i].bottom),cv::Scalar(71, 99, 250),2);
						std::stringstream ss;  
						ss << CLASSES[(int)(objects[k].boxs[i].classid)] << "/" << objects[k].boxs[i].confidence;  
						std::string  text = ss.str();  
						cv::putText(objects[k].orgimg, text, cvPoint(objects[k].boxs[i].left,objects[k].boxs[i].top+20), cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 255, 255));  	
					}
				}		
				if(fpshaswrite[objects[k].inputid]==0){
					std::stringstream ss;
					ss << "FPS: " << each_fps[objects[k].inputid] << "/" << total_fps; 
					std::string  text = ss.str();  				
					cv::putText(objects[k].orgimg, text, cvPoint(0,20), cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(127, 255, 0));  	
					objects[k].orgimg.copyTo(eachscreen[objects[k].inputid]);
					fpshaswrite[objects[k].inputid]=1;
				}
			}	
			if(gdisplaytype==XCBSHOW)
				XCBShow::Instance().imshow(0,onescreen);
			//else if(gdisplaytype==SDLSHOW)
			//	SDL2Show::Instance().imshow(0,onescreen);
			else{
				cv::imshow("input0",onescreen);
				cv::waitKey(1);
			}
		}		
	}
	if(gdisplaytype==OCVIMGSHOW)
		cv::destroyAllWindows();
	return NULL;
}

void *thr_fps(void *arg)
{
	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
	typedef std::chrono::duration<float> fsec;	

	safesleep(1000);//1000ms to fullfill buffer
	total_frame = 0;
	memset(each_frame,0,sizeof(each_frame));	
	auto t0=Time::now(),t1=Time::now();
	while(grunning){
		safesleep(1000*2);//2s
		t1 = Time::now();
    fsec fs = t1 - t0;
		double timeUsed = std::chrono::duration_cast<ms>(fs).count();
		total_fps = total_frame*1000/timeUsed;
		for(int i=0;i<sizeof(each_frame)/sizeof(each_frame[0]);i++){
			each_fps[i] = each_frame[i]*1000/timeUsed;
		}
		total_frame = 0;
		memset(each_frame,0,sizeof(each_frame));		
		std::cout << "Cur fps=" << total_fps << "\n";
		t0 = Time::now();
	}
	return NULL;
}

int main(int argc, char** argv) {

	int i;
	bool bcamera=true;
	if (argc < 3) {
		std::cout << "mobilenet_ssd model_prototxt model_weights <--(call sudo init 3 will get best performance)\n";
		std::cout << "mobilenet_ssd model_prototxt model_weights 1  <--(if you want display, just add 1)\n";
		std::cout << "mobilenet_ssd model_prototxt model_weights 1 videopath <--(will dec 6 video files in video folder)\n";
		std::cout << "mobilenet_ssd model_prototxt model_weights 0 videopath <--(will dec 6 video files in video folder without display)\n";
		std::cout << "mobilenet_ssd model_prototxt model_weights 1 videopath hw<--(will dec 6 video files in video folder and with hardware dec)\n";
		//std::cout << "mobilenet_ssd model_prototxt model_weights 1sdl ***<--(1sdl mean refresh with sdl2)\n";
		std::cout << "mobilenet_ssd model_prototxt model_weights 1xcb ***<--(1xcb mean refresh with xcb)\n";
		return 1;
	}
	const string& model_file = argv[1];
	const string& weights_file = argv[2];
	if(argc>=4){
		isshow = argv[3][0]-'0';
		if(argv[3][1]=='s' || argv[3][1]=='S')
			gdisplaytype = SDLSHOW;
		else if(argv[3][1]=='x' || argv[3][1]=='X') 
			gdisplaytype = XCBSHOW;
		else
			gdisplaytype = OCVIMGSHOW;
	}
	if(argc>=5)
		bcamera=false;
	if(argc>=6)
		bhardware_dec=true;
	
	grunning = true;
	total_frame = 0;
	total_fps = 0;
	memset(each_frame,0,sizeof(each_frame));
	memset(each_fps,0,sizeof(each_fps));
	pthread_t nintid;
	pthread_t nrtid;
	pthread_t nfpstid;
	sem_init(&g_semt, 0, 0);
	
	//std::cout << "Opencv is using Opencl? " << cv::ocl::useOpenCL() << "\n";
	//cv::ocl::setUseOpenCL(false);
	//std::cout << "Now set to " << cv::ocl::useOpenCL() << "\n";
	// Initialize the network.
#ifdef RUNWITH_SHOW
	pthread_t nshowtid;
	if(isshow){
		sem_init(&g_semtshow, 0, 0);
		pthread_mutex_init(&mutexshow,NULL); 	
		pthread_create(&nshowtid, NULL, thr_show, NULL);	
	}
	Detector detector(model_file, weights_file,isshow);	
#else
	Detector detector(model_file, weights_file,false);	
#endif
	gpdetector = &detector;
	pthread_create(&nrtid, NULL, thr_detector, (void*)(&detector));
	bindcore(nrtid,0);
	pthread_create(&nfpstid, NULL, thr_fps, NULL);
	if(bcamera)
		pthread_create(&nintid, NULL, thr_camera, NULL);
	else
		pthread_create(&nintid, NULL, thr_video, (void*)(argv[4]));
	bindcore(nintid,2);
	
	//wait quit
	while(true){
		std::cout << "Enter q to quite: " ;
		int c=getchar();
		if (c=='q' || c=='Q'){
			grunning=false;
			gpdetector->Stop();
			safewakeup();
			break;
		}
	}
	pthread_join(nintid,NULL);
	pthread_join(nrtid,NULL);
	pthread_cancel(nfpstid);
	pthread_join(nfpstid,NULL);
#ifdef RUNWITH_SHOW
	if(isshow){
		pthread_join(nshowtid,NULL);
		sem_destroy(&g_semtshow);
		pthread_mutex_destroy(&mutexshow);  	
	}
#endif
	sem_destroy(&g_semt);

	std::cout<< "Done";
	return 0;
}

