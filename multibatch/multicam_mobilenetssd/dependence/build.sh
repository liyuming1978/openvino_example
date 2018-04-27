#!/bin/bash
DIR="$( cd "$( dirname "$0"  )" && pwd  )"
sudo apt-get install autoconf automake libtool yasm ffmpeg
sudo apt-get install libxcb-composite0-dev libxcb-glx0-dev libxcb-dri2-0-dev libxcb-xf86dri0-dev libxcb-xinerama0-dev libxcb-render-util0-dev libxcb-xv0-dev

git clone https://chromium.googlesource.com/libyuv/libyuv

sudo mkdir -p /opt/libyuv && sudo chmod a+w /opt/libyuv

cd $DIR/libyuv
git clean -dxf
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="/opt/libyuv" -DCMAKE_BUILD_TYPE="Release" ..
cmake --build . --config Release
cmake --build . --target install --config Release
