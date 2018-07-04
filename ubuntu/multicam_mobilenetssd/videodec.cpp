#include "videodec.hpp"
#include "libyuv.h"

VideoDec::VideoDec() {
	// Register all formats and codecs
	av_register_all();	
	/// Allocate video frame
	pFrame = av_frame_alloc();
}

VideoDec::~VideoDec() {
	close();
	av_frame_free(&pFrame);
}

void VideoDec::InitLoad(const string& video_file,int id,bool bhwl) {
	_inputfilename = video_file;
	_isopen = false;
	_id = id;
	_hardware_accl = bhwl;
}

cv::Mat VideoDec::GetFrame(int width,int height) {
	for(int i=0;i<2;i++) {
		if(open()) {
			AVPacket        packet;
			int             got_frame=0;
			av_init_packet(&packet);
			
			while(av_read_frame(pFormatCtx, &packet) >= 0) {
				// Is this a packet from the video stream?
				if(packet.stream_index==videoStreamIdx) {
					/// Decode video frame
					avcodec_decode_video2(pCodecCtx, pFrame, &got_frame, &packet);
					// Did we get a video frame?
					if(got_frame) {
						//printf("Frame [%d]: pts=%lld, pkt_pts=%lld, pkt_dts=%lld\n", pFrame->format, pFrame->pts, pFrame->pkt_pts, pFrame->pkt_dts);
						//std::cout << pFrame->format << "\n";
						//printf("Frame [%d]: pts=%d, pkt_pts=%d, pkt_dts=%d\n", pFrame->format, pFrame->linesize[0], pFrame->linesize[1], pFrame->linesize[2]);
						//uint8* dst_frame= new uint8[height*width*3]; -- if create Mat with new point, will not auto-delete
						cv::Mat ret(height, width, CV_8UC3);
						libyuv::I420ToRGB24(pFrame->data[0],pFrame->linesize[0],
											pFrame->data[1],pFrame->linesize[1],
											pFrame->data[2],pFrame->linesize[2],
											ret.data,width*3,width,height);
						av_frame_unref(pFrame);
						av_free_packet(&packet);
						return ret;
					}
				}
				// Free the packet that was allocated by av_read_frame
				av_free_packet(&packet);
			}
			if(!got_frame) {  //play end, just loop
				/*auto stream = pFormatCtx->streams[videoStreamIdx];
				avio_seek(pFormatCtx->pb, 0, SEEK_SET);
				avformat_seek_file(pFormatCtx, videoStreamIdx, 0, 0, stream->duration, 0);*/
				close();
				std::cout << "repeat";
			}
		}
		else
			break;
	}
	return  cv::Mat::zeros(height,width,CV_8UC3);
}

bool VideoDec::open() {
	if(!_isopen) {
		pFormatCtx = NULL;
		/// Open video file
    if(avformat_open_input(&pFormatCtx, _inputfilename.c_str(), 0, NULL) != 0) {
			av_free(pFormatCtx);
			return false;
		}
    /// Retrieve stream information
    if(avformat_find_stream_info(pFormatCtx, NULL) < 0) {
			av_free(pFormatCtx);
			return false;			
		}
    /// Find the first video stream
		videoStreamIdx=-1;
		for(int i=0; i< pFormatCtx->nb_streams; i++)
		{
				if(pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) { //CODEC_TYPE_VIDEO
						videoStreamIdx=i;
						break;
				}
		}
		if(videoStreamIdx<0) {
			av_free(pFormatCtx);
			return false;			
		}
    /// Get a pointer to the codec context for the video stream and Find the decoder for the video stream
		pCodecCtx = pFormatCtx->streams[videoStreamIdx]->codec;
		if(_hardware_accl){
			//pCodec = avcodec_find_decoder(pCodecCtx->codec_id); 
			pCodec = avcodec_find_decoder_by_name("libyami_h264"); //hardware decoder
			pCodecCtx->coder_type = 0;
		}
		else
			pCodec = avcodec_find_decoder_by_name("h264"); //software decoder
    if(pCodec==NULL) {
			av_free(pFormatCtx);
			//printf("none codec");
			return false;		
    }
    /// Open codec
    //pCodecCtx->thread_count=8;
		//pCodecCtx->thread_type = FF_THREAD_FRAME;
    if( avcodec_open2(pCodecCtx, pCodec, NULL) < 0 ) {
			av_free(pFormatCtx);
			return false;		
		}
		_isopen = true;
	}
	return true;
}

void VideoDec::close() {
	if(_isopen) {
		avcodec_close(pCodecCtx);
		avformat_close_input(&pFormatCtx);
		av_free(pFormatCtx);
		_isopen = false;
	}
}
