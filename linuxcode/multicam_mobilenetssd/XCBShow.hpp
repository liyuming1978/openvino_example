#ifndef __XCBSHOW_HPP_
#define __XCBSHOW_HPP_
//note! not thread-safe!!!
#define SDLSHOW_MAXID 50
#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <xcb/xcb.h>
#include <xcb/shm.h>
#include <xcb/xv.h>

class XCBShow {
public:
	static XCBShow& Instance() {
			static XCBShow theXCBShow;
			return theXCBShow;
	}
	/* more (non-static) functions here */
	void imshow(int id,cv::Mat const &img);

private:
	typedef struct __xcbwin {
		xcb_connection_t * conn;
		xcb_window_t window;
		xcb_gcontext_t gc;  
		cv::Size wh;
	}xcbwin;
	XCBShow();
	XCBShow(XCBShow const&){}              // copy ctor hidden
	~XCBShow();
	void destory(int id);
	void create_win(int id);
	xcbwin _xcbwins[SDLSHOW_MAXID];
	bool binit;
	xcb_xv_format_t* fvisual;
	uint32_t fid;
	xcb_xv_port_t fport;
};

#endif //USE_OPENCV
#endif //__XCBSHOW_HPP_