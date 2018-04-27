#ifndef __VIDEODEC_HPP_
#define __VIDEODEC_HPP_
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
//#include <swscale.h>  -- libyuv
//#include <libavdevice/avdevice.h>
}
using std::string;

class VideoDec {
public:
	VideoDec();
	~VideoDec();
	cv::Mat GetFrame(int width,int height);   //no wait
	void InitLoad(const string& video_file,int id,bool bhwl);
	inline int GetID(){return _id;}

private:
	bool open();
	void close();
	string _inputfilename;
	bool _isopen;
  AVFormatContext *pFormatCtx;
  AVCodecContext  *pCodecCtx;
  AVCodec         *pCodec;
	AVFrame        *pFrame;		
	int videoStreamIdx;
	int _id;
	bool _hardware_accl;
};
#endif  // USE_OPENCV
#endif //__VIDEODEC_HPP_