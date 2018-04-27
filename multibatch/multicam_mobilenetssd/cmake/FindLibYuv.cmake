### FindLIBYUV.cmake ---
## 
## Author: Julien Wintz
## Created: Mon Oct 21 10:21:28 2013 (+0200)
## Version: 
## Last-Updated: Mon Oct 21 10:47:14 2013 (+0200)
##           By: Julien Wintz
##     Update #: 55
######################################################################
## 
### Change Log: change to libyuv by yumingli
## 
######################################################################

# - Try to find LIBYUV
# Once done this will define
#
#  LIBYUV_FOUND - system has LIBYUV
#  LIBYUV_INCLUDE_DIRS - the LIBYUV include directory
#  LIBYUV_LIBRARIES - Link these to use LIBYUV
#  LIBYUV_DEFINITIONS - Compiler switches required for using LIBYUV
#
#  Copyright (c) 2008 Andreas Schneider <mail@cynapses.org>
#  Modified for other libraries by Lasse Kärkkäinen <tronic>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.

include(FindPackageHandleStandardArgs)
include(GetPrerequisites)

find_path(LIBYUV_INCLUDE_DIR
	NAMES libyuv.h
	PATHS /opt/libyuv/include /usr/include /usr/local/include /opt/local/include)

find_library(LIBYUV_LIBRARY
	NAMES yuv
	PATHS /opt/libyuv/lib /usr/lib /usr/local/lib /opt/local/lib)

if(LIBYUV_LIBRARY)
	set(LIBYUV_FOUND TRUE)
endif(LIBYUV_LIBRARY)

mark_as_advanced(LIBYUV_INCLUDE_DIR LIBYUV_LIBRARY)
