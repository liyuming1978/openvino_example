#include "detector.hpp"
#include "videodec.hpp"
//#include "SDL2Show.hpp"
#include "XCBShow.hpp"
#include "landmark.h"
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
bool gshowmark=false;
Detector* gpdetector;
sem_t g_semt;
int total_frame;
int each_frame[INPUTNUM];
float total_fps;
float each_fps[INPUTNUM];
sem_t g_semtshow;
queue<vector<Detector::DetctorResult> > gresultque;
pthread_mutex_t mutexshow ; 
pthread_mutex_t mutexcamimg; 
int isshow=0;
bool bhardware_dec=false;
enum DISPLAY_TYPE {
	NONESHOW=-1,
	OCVIMGSHOW=0,
	SDLSHOW,
	XCBSHOW,
}gdisplaytype;
static string  CLASSES[] = {"background",
           "face", "bicycle", "bird", "boat",
           "!bottle", "!bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "!pottedplant",
						"sheep", "sofa", "!train", "!tvmonitor"};

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
	char* videopath = (char*)arg;
	string videoname=videopath;
	videoname+="/";
	VideoDec vdec[INPUTNUM];
	
	for(int i=0;i<INPUTNUM;i++) {
		vdec[i].InitLoad(videoname+std::to_string(i%6)+".mp4",i,bhardware_dec);
	}	
	//dec each with 28fps
	int curbatch=INPUTNUM;
	Detector::InsertImgStatus faceret=Detector::INSERTIMG_NULL;
	vector<Detector::DetctorResult> objects;
	while (grunning){
		for(int i=0;i<curbatch;i++) {
			//note , here I do each one by one, so avoid heavy IO, and do as quick as possible, no wait
			//cv::Mat frame = vdec[i].GetFrame(gpdetector->GetNetSize().width,gpdetector->GetNetSize().height);
			cv::Mat frame = vdec[i].GetFrame(ginput_width,ginput_height);
			faceret = gpdetector->InsertImage(frame,objects,i);		
			if (Detector::INSERTIMG_GET == faceret ||Detector::INSERTIMG_PROCESSED == faceret) {  //aSync call, you must use the ret image
				for(int k=0;k<objects.size();k++){
					each_frame[objects[k].inputid]+=1;
					total_frame++;	
				}	
				if(isshow && Detector::INSERTIMG_GET == faceret){
					pthread_mutex_lock(&mutexshow); 	
					gresultque.push(objects);
					pthread_mutex_unlock(&mutexshow); 	
					sem_post(&g_semtshow);
					
					pthread_mutex_lock(&mutexshow); 
					while(gresultque.size()>=2 && grunning){  //only cache 2 batch
						pthread_mutex_unlock(&mutexshow); 	
						usleep(1*1000); //sleep 2ms to recheck
						pthread_mutex_lock(&mutexshow); 
					}			
					pthread_mutex_unlock(&mutexshow); 					
				}				
			}
		}
	}

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

string lmodel_file="";
string lweights_file="";
void *thr_show(void *arg)
{
	vector<cv::Mat> landmarkimg;
	vector<Detector::resultbox> landmarkrt;
	LandMark lm4;
	LandMark lm8;
	if(gshowmark){
		lm4.Load(lmodel_file, lweights_file, 4);
		lm8.Load(lmodel_file, lweights_file, 8);
	}
	//---load---------------
	bool hasmove=false;
	int fpshaswrite[INPUTNUM];
	cv::Mat onescreen(ginput_height*2, ginput_width*3, CV_8UC3);
	cv::Mat eachscreen[INPUTNUM];
	for(int i=0;i<INPUTNUM;i++)
		eachscreen[i]=onescreen(cv::Rect(ginput_width*(i%3),(i>2)?ginput_height:0,ginput_width,ginput_height));
	
	while(grunning){
		sem_wait(&g_semtshow);
		vector<Detector::DetctorResult> objects;
		if(!gresultque.empty()){
			pthread_mutex_lock(&mutexshow); 	
			objects = gresultque.front();	
			gresultque.pop();			
			pthread_mutex_unlock(&mutexshow); 	
			memset(fpshaswrite,0,sizeof(fpshaswrite));
			
			if(gshowmark)
			{
				for(int k=0;k<objects.size();k++){
					for(int i=0;i<objects[k].boxs.size();i++){
						if(objects[k].boxs[i].right-objects[k].boxs[i].left>30){
							landmarkimg.push_back(objects[k].orgimg);
							landmarkrt.push_back(objects[k].boxs[i]);
						}				
					}
				}
				//std::cout <<"lym "<<landmarkimg.size()<<"\n";
				int tl8=landmarkimg.size()/8*8;
				for(int ll=0;ll<tl8;ll+=8){
					for (int lj=0;lj<8;lj++){
						if(ll+lj<landmarkimg.size()){
							lm8.insertImage(landmarkimg[ll+lj].clone()(cv::Rect(landmarkrt[ll+lj].left,landmarkrt[ll+lj].top,
								landmarkrt[ll+lj].right-landmarkrt[ll+lj].left+1,landmarkrt[ll+lj].bottom-landmarkrt[ll+lj].top+1)));
						}
					}
					float* plandmark = lm8.getLandmark();
					for (int lj=0;lj<8;lj++){
						if(ll+lj<landmarkimg.size()){
							for (int j = 0; j < LANDMARK_COUNT / 2; j++) {
								plandmark[lj*LANDMARK_COUNT+2 * j] = plandmark[lj*LANDMARK_COUNT+2 * j] * (landmarkrt[ll+lj].right - landmarkrt[ll+lj].left + 1) / 128 + landmarkrt[ll+lj].left;
								plandmark[lj*LANDMARK_COUNT+2 * j + 1] = plandmark[lj*LANDMARK_COUNT+2 * j + 1] * (landmarkrt[ll+lj].bottom - landmarkrt[ll+lj].top + 1) / 128 + landmarkrt[ll+lj].top;
								cv::circle(landmarkimg[ll+lj], cv::Point((int)plandmark[lj*LANDMARK_COUNT+2 * j], (int)plandmark[lj*LANDMARK_COUNT+2 * j + 1]), 1, cv::Scalar(0, 255, 0), 2);
							}					
						}
					}
				}
				int tl4=(landmarkimg.size()-tl8+3)/4*4;
				for(int ll=0;ll<tl4;ll+=4){
					for (int lj=0;lj<4;lj++){
						if(tl8+ll+lj < landmarkimg.size()){
							lm4.insertImage(landmarkimg[tl8+ll+lj].clone()(cv::Rect(landmarkrt[tl8+ll+lj].left,landmarkrt[tl8+ll+lj].top,
								landmarkrt[tl8+ll+lj].right-landmarkrt[tl8+ll+lj].left+1,landmarkrt[tl8+ll+lj].bottom-landmarkrt[tl8+ll+lj].top+1)));
						}
					}
					float* plandmark = lm4.getLandmark();
					for (int lj=0;lj<4;lj++){
						if(tl8+ll+lj<landmarkimg.size()){
							for (int j = 0; j < LANDMARK_COUNT / 2; j++) {
								plandmark[lj*LANDMARK_COUNT+2 * j] = plandmark[lj*LANDMARK_COUNT+2 * j] * (landmarkrt[tl8+ll+lj].right - landmarkrt[tl8+ll+lj].left + 1) / 128 + landmarkrt[tl8+ll+lj].left;
								plandmark[lj*LANDMARK_COUNT+2 * j + 1] = plandmark[lj*LANDMARK_COUNT+2 * j + 1] * (landmarkrt[tl8+ll+lj].bottom - landmarkrt[tl8+ll+lj].top + 1) / 128 + landmarkrt[tl8+ll+lj].top;
								cv::circle(landmarkimg[tl8+ll+lj], cv::Point((int)plandmark[lj*LANDMARK_COUNT+2 * j], (int)plandmark[lj*LANDMARK_COUNT+2 * j + 1]), 1, cv::Scalar(0, 255, 0), 2);
							}					
						}
					}				
				}			
				landmarkimg.clear();
				landmarkrt.clear();
			}
			
			for(int k=0;k<objects.size();k++){
				for(int i=0;i<objects[k].boxs.size();i++){
					if(CLASSES[(int)(objects[k].boxs[i].classid)][0]=='!')
						continue;
					cv::rectangle(objects[k].orgimg,cvPoint(objects[k].boxs[i].left,objects[k].boxs[i].top),cvPoint(objects[k].boxs[i].right,objects[k].boxs[i].bottom),cv::Scalar(71, 99, 250),2);
					std::stringstream ss;  
					ss << CLASSES[(int)(objects[k].boxs[i].classid)] << "/" << objects[k].boxs[i].confidence;  
					std::string  text = ss.str();  
					cv::putText(objects[k].orgimg, text, cvPoint(objects[k].boxs[i].left,objects[k].boxs[i].top+20), cv::FONT_HERSHEY_PLAIN, 1.0f, cv::Scalar(0, 255, 255));  	
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
			else if(gdisplaytype==OCVIMGSHOW){
				cv::imshow("input0",onescreen);
				cv::waitKey(1);
			}
		}		
	}
	if(gdisplaytype==OCVIMGSHOW)
		cv::destroyAllWindows();
	return NULL;
}

int main(int argc, char** argv) {

	int i;
	if (argc < 4) {
		std::cout << "mobilenet_ssd model_prototxt model_weights 1 videopath <--(will dec 6 video files in video folder)\n";
		std::cout << "mobilenet_ssd model_prototxt model_weights 0 videopath <--(will dec 6 video files in video folder without display)\n";
		//std::cout << "mobilenet_ssd model_prototxt model_weights 1 videopath hw<--(will dec 6 video files in video folder and with hardware dec)\n";
		//std::cout << "mobilenet_ssd model_prototxt model_weights 1sdl ***<--(1sdl mean refresh with sdl2)\n";
		std::cout << "mobilenet_ssd model_prototxt model_weights 1xcb videopath<--(1xcb mean refresh with xcb)\n";
		return 1;
	}
	
	const string& model_file = argv[1];
	const string& weights_file = argv[2];
	gdisplaytype = NONESHOW;
	if(argc>=4){
		isshow = argv[3][0]-'0';
		if(argv[3][1]=='s' || argv[3][1]=='S')
			gdisplaytype = SDLSHOW;
		else if(argv[3][1]=='x' || argv[3][1]=='X') 
			gdisplaytype = XCBSHOW;
		else if(isshow>0)
			gdisplaytype = OCVIMGSHOW;
	}
	if(argc>=6) {
		gshowmark = true;
		lmodel_file = argv[5];
		lweights_file = argv[6];	
	}
	
	Detector facedet;
	gpdetector = &facedet;
	gpdetector->Load(model_file, weights_file,6);
	
	grunning = true;
	total_frame = 0;
	total_fps = 0;
	memset(each_frame,0,sizeof(each_frame));
	memset(each_fps,0,sizeof(each_fps));

	pthread_t nfpstid;
	pthread_t nintid;
	pthread_t nshowtid;
	if(isshow){
		sem_init(&g_semtshow, 0, 0);
		pthread_mutex_init(&mutexshow,NULL); 	
		pthread_create(&nshowtid, NULL, thr_show, NULL);	
	}	
	pthread_create(&nintid, NULL, thr_video, (void*)(argv[4]));
	pthread_create(&nfpstid, NULL, thr_fps, NULL);	
	//bindcore(nintid,2);
	
	//wait quit
	while(true){
		std::cout << "Enter q to quite: " ;
		int c=getchar();
		if (c=='q' || c=='Q'){
			grunning=false;
			break;
		}
	}
	pthread_join(nintid,NULL);
	pthread_cancel(nfpstid);
	pthread_join(nfpstid,NULL);
	if(isshow){
		pthread_join(nshowtid,NULL);
		sem_destroy(&g_semtshow);
		pthread_mutex_destroy(&mutexshow);  	
	}
	
	std::cout<< "Done";
	return 0;
}

