@REM Copyright (c) 2018 Intel Corporation

@REM Licensed under the Apache License, Version 2.0 (the "License");
@REM you may not use this file except in compliance with the License.
@REM You may obtain a copy of the License at

@REM      http://www.apache.org/licenses/LICENSE-2.0

@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS,
@REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM See the License for the specific language governing permissions and
@REM limitations under the License.


@setlocal
@echo off

set "ROOT_DIR=%~dp0"

set "SOLUTION_DIR64=%ROOT_DIR%\build"
set "InferenceEngine_DIR=C:\Intel\computer_vision_sdk_2018.2.298\deployment_tools\inference_engine\share"
set "OpenCV_DIR=C:\Intel\computer_vision_sdk_2018.2.298\opencv"


echo Creating Visual Studio 2015 (x64) files in %SOLUTION_DIR64%... && ^
cd "%ROOT_DIR%" && cmake -E make_directory "%SOLUTION_DIR64%" && cd "%SOLUTION_DIR64%" && cmake -G "Visual Studio 14 2015 Win64" "%ROOT_DIR%"

echo Done.
pause