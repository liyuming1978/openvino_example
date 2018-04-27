#include "XCBShow.hpp"
#include "detector.hpp"
#include <X11/Xatom.h> /* XA_WM_NAME */
#include "libyuv.h"

//XCB  (vlc change settings to all to see it's xcb, XVideo output (XCB)) 
//-- https://github.com/videolan/vlc/blob/a9712a7e4fba854fceab8ce5f232b6baf05dc39c/modules/video_output/xcb/xvideo.c
//-- https://github.com/mstorsjo/vlc/blob/master/modules/video_output/xcb/pictures.c
//apt-get install libxcb-composite0-dev libxcb-glx0-dev libxcb-dri2-0-dev libxcb-xf86dri0-dev libxcb-xinerama0-dev libxcb-render-util0-dev libxcb-xv0-dev
//xvinfo to check 
//https://rosettacode.org/wiki/Window_creation/X11
//https://xcb.freedesktop.org/tutorial/basicwindowsanddrawing/
//https://github.com/huceke/xine-lib-vaapi/blob/master/src/video_out/video_out_xcbxv.c
XCBShow::XCBShow(){
	memset(_xcbwins,0,sizeof(_xcbwins));
	binit = false;
	xcb_connection_t *conn = xcb_connect (NULL, NULL);
  xcb_xv_query_extension_reply_t *r;
  xcb_xv_query_extension_cookie_t ck = xcb_xv_query_extension (conn);	
	// We need XVideo 2.2 for PutImage
	r = xcb_xv_query_extension_reply (conn, ck, NULL);
	if (r == NULL){
		std::cout<< "XVideo extension not available";
		xcb_disconnect(conn);
		return;
	}
	else{
		if (r->major != 2 || r->minor < 2)
			std::cout<< "XVideo extension version is not 2.2";
		else 
			binit = true;
		free (r);	
	}
	if(!binit){
		xcb_disconnect(conn);
		return;
	}
	
	/* Get the first screen */
	const xcb_setup_t      *setup  = xcb_get_setup (conn);
	xcb_screen_iterator_t   iter   = xcb_setup_roots_iterator (setup);
	xcb_screen_t           *screen = iter.data;
	
	/* Cache adaptors infos */
	fvisual=NULL;
	fport=0;
	xcb_xv_query_adaptors_reply_t *adaptors =
			xcb_xv_query_adaptors_reply (conn,
					xcb_xv_query_adaptors (conn, screen->root), NULL);
	if (adaptors == NULL){
		binit = false;
		xcb_disconnect(conn);
		return;		
	}
	xcb_xv_adaptor_info_iterator_t it;
	for (it = xcb_xv_query_adaptors_info_iterator (adaptors);
			 it.rem > 0;
			 xcb_xv_adaptor_info_next (&it))
	{
		const xcb_xv_adaptor_info_t *a = it.data;
		if (!(a->type & XCB_XV_TYPE_INPUT_MASK)
		 || !(a->type & XCB_XV_TYPE_IMAGE_MASK))
				continue;

		/* Look for an RGB image format */
		bool bfound = false;
		xcb_xv_list_image_formats_reply_t *list =
			xcb_xv_list_image_formats_reply (conn,
					xcb_xv_list_image_formats (conn, a->base_id), NULL);
		if (list == NULL)
				continue;
		xcb_xv_format_t *formats = xcb_xv_adaptor_info_formats (a);
		for (const xcb_xv_image_format_info_t *f =
					 xcb_xv_list_image_formats_format (list),
																				*f_end =
					 f + xcb_xv_list_image_formats_format_length (list);
					 f < f_end; f++) {
				 if (f->type == XCB_XV_IMAGE_FORMAT_INFO_TYPE_YUV){
					 bfound = true;
					 fid=f->id;
					 break;
				 }
				 formats++;
		}
		if(!bfound) continue;
			 
		bfound = false;
		/* Grab a port */
		for (unsigned i = 0; i < a->num_ports; i++)
		{
				 fport = a->base_id + i;
				 xcb_xv_grab_port_reply_t *gr =
						 xcb_xv_grab_port_reply (conn,
								 xcb_xv_grab_port (conn, fport, XCB_CURRENT_TIME), NULL);
				 uint8_t result = gr ? gr->result : 0xff;

				 free (gr);
				 if (result == 0)
				 {
						 bfound = true;
						 break;
				 }
				 std::cout<< "cannot grab port " << fport;
		}
		if(!bfound) continue; /* No usable port */
		fvisual = formats;
	}
	if(fvisual==NULL){
		binit = false;
		if(fport>0)
			xcb_xv_ungrab_port (conn, fport, XCB_CURRENT_TIME);
	}	
	xcb_disconnect(conn);
}        
	
XCBShow::~XCBShow(){
	for(int i=0;i<SDLSHOW_MAXID;i++){
		destory(i);
	}
}

void XCBShow::destory(int id){
	if(_xcbwins[id].conn!=NULL)
		xcb_disconnect(_xcbwins[id].conn);
	memset(&_xcbwins[id],0,sizeof(xcbwin));
}

void XCBShow::create_win(int id){
	std::string title = "input"+std::to_string(id);

	xcb_connection_t *conn = xcb_connect (NULL, NULL);
	/* Get the first screen */
	const xcb_setup_t      *setup  = xcb_get_setup (conn);
	xcb_screen_iterator_t   iter   = xcb_setup_roots_iterator (setup);
	xcb_screen_t           *screen = iter.data;
	
	xcb_pixmap_t pixmap = xcb_generate_id (conn);
	/* Create the window */
	uint32_t mask =
			XCB_CW_BACK_PIXMAP |
			XCB_CW_BACK_PIXEL |
			XCB_CW_BORDER_PIXMAP |
			XCB_CW_BORDER_PIXEL |
			XCB_CW_COLORMAP;
	uint32_t values[] = {
			/* XCB_CW_BACK_PIXMAP */
			pixmap,
			/* XCB_CW_BACK_PIXEL */
			screen->black_pixel,
			/* XCB_CW_BORDER_PIXMAP */
			pixmap,
			/* XCB_CW_BORDER_PIXEL */
			screen->black_pixel,
			/* XCB_CW_COLORMAP */
			screen->default_colormap,
	};
	xcb_window_t window = xcb_generate_id (conn);
	xcb_void_cookie_t ck;
	xcb_generic_error_t *err;
	xcb_create_pixmap (conn, screen->root_depth, pixmap, screen->root, 1, 1);
	ck = xcb_create_window_checked  (conn,                    /* Connection          */
										 screen->root_depth,          /* depth (same as root)*/
										 window,                        /* window Id           */
										 screen->root,                  /* parent window       */
										 0,0,                          /* x, y                */
										 _xcbwins[id].wh.width, _xcbwins[id].wh.height,                      /* width, height       */
										 0,                            /* border_width        */
										 XCB_WINDOW_CLASS_INPUT_OUTPUT, /* class               */
										 fvisual->visual,           		/* visual              */
										 mask, values);                 /* masks, values */
	err = xcb_request_check (conn, ck);
	if (err)
	{
			std::cout<< "creating window: X11 error " << err->error_code;
			free (err);
			xcb_xv_ungrab_port (conn, fport, XCB_CURRENT_TIME);
			binit = false;
			return;
	}				
	xcb_change_property (conn, XCB_PROP_MODE_REPLACE, window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING,
											 /* format */ 8, strlen (title.c_str()), title.c_str());
	xcb_map_window (conn, window);	
	xcb_flush (conn);
	_xcbwins[id].gc = xcb_generate_id (conn);
	xcb_create_gc (conn, _xcbwins[id].gc, window, 0, NULL);	
	

	_xcbwins[id].conn = conn;
	_xcbwins[id].window = window;	
}

void XCBShow::imshow(int id,cv::Mat const &img) {
	if(id>=SDLSHOW_MAXID||!binit)
		return;
	if(_xcbwins[id].wh!=img.size())
		destory(id);
	if(_xcbwins[id].conn==NULL){
		_xcbwins[id].wh = img.size();
		create_win(id);
		if(_xcbwins[id].conn==NULL)
			return;
	}
	
  /*xcb_xv_query_image_attributes_cookie_t query_attributes_cookie;
  xcb_xv_query_image_attributes_reply_t *query_attributes_reply;
  query_attributes_cookie = xcb_xv_query_image_attributes(_xcbwins[id].conn, fport, fid, 640, 480);
  query_attributes_reply = xcb_xv_query_image_attributes_reply(_xcbwins[id].conn, query_attributes_cookie, NULL);	
	uint32_t *a = xcb_xv_query_image_attributes_pitches (query_attributes_reply);
  std::cout << query_attributes_reply->data_size <<"\n";
  std::cout << query_attributes_reply->width <<"\n";
  std::cout << query_attributes_reply->height <<"\n";	
	std::cout << a[0] <<"\n" << a[1] <<"\n";	*/
	//display data
	int width = img.size().width;
	int height = img.size().height;
	uint8* pARGB = new uint8[width*height*4];
	uint8* pYUY2 = new uint8[width*height*2];
	libyuv::RGB24ToARGB(img.data,width*3,
										pARGB,width*4,
										width,height
	);	
						
	libyuv::ARGBToYUY2(pARGB,width*4,
										pYUY2,width*2,
										width,height
	);	
	delete pARGB;
	
	xcb_void_cookie_t ck;
	ck=xcb_xv_put_image_checked(_xcbwins[id].conn, fport, _xcbwins[id].window,
										_xcbwins[id].gc, fid,
										0, 0, width,height,
										0,0, width, height,
										width,height,
										width*height*2, pYUY2);
	/* Wait for reply. See x11.c for rationale. */
	xcb_generic_error_t *e = xcb_request_check (_xcbwins[id].conn, ck);
	if (e != NULL)
	{
		std::cout <<  e->error_code <<"\n";
		free (e);
	}					
	delete pYUY2;
}

