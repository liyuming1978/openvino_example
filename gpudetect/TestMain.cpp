/////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");// you may not use this file except in compliance with the License.// You may obtain a copy of the License at//// http://www.apache.org/licenses/LICENSE-2.0//// Unless required by applicable law or agreed to in writing, software// distributed under the License is distributed on an "AS IS" BASIS,// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.// See the License for the specific language governing permissions and// limitations under the License.
/////////////////////////////////////////////////////////////////////////////////////////////

#include <conio.h>
#include <stdio.h>
#include "DeviceId.h"
#include <string>

#include "D3D11.h"
#include "ID3D10Extensions.h"
#include <intrin.h>


// Define settings to reflect Fidelity abstraction levels you need
typedef enum
{
	NotCompatible,  // Found GPU is not compatible with the app
	Low,
	Medium,
	MediumPlus,
	High,
	Undefined  // No predefined setting found in cfg file. 
			   // Use a default level for unknown video cards.
}
PresetLevel;

const char *PRODUCT_FAMILY_STRING[] =
{
	"Sandy Bridge",
	"Ivy Bridge",
	"Haswell",
	"ValleyView",
	"Broadwell",
	"Cherryview",
	"Skylake",
	"Kabylake",
	"Coffeelake"
};
const unsigned int NUM_PRODUCT_FAMILIES = sizeof(PRODUCT_FAMILY_STRING) / sizeof(PRODUCT_FAMILY_STRING[0]);

PresetLevel getDefaultFidelityPresets(unsigned int VendorId, unsigned int DeviceId);

bool IsGPUOK()
{
	//
	// Some information about the gfx adapters is exposed through Windows and DXGI. If
	// the machine has multiple gfx adapters or no Intel gfx this is where that can be
	// detected.
	//
	unsigned int VendorId, DeviceId;
	unsigned __int64 VideoMemory;
	std::wstring GFXBrand;

	if (false == getGraphicsDeviceInfo(&VendorId, &DeviceId, &VideoMemory, &GFXBrand))
	{
		printf("Intel Graphics Adapter not detected\n");
		return false;
	}

	//
	// Check for DX extensions
	//
	unsigned int extensionVersion = checkDxExtensionVersion();
	if (extensionVersion >= ID3D10::EXTENSION_INTERFACE_VERSION_1_0) {
	}
	else {
		return false;
	}

	//
	// In DirectX, Intel exposes additional information through the driver that can be obtained
	// querying a special DX counter
	//
	IntelDeviceInfoHeader intelDeviceInfoHeader = { 0 };
	unsigned char intelDeviceInfoBuffer[1024];

	long getStatus = getIntelDeviceInfo(VendorId, &intelDeviceInfoHeader, &intelDeviceInfoBuffer);

	if (getStatus == GGF_SUCCESS)
	{
		if (intelDeviceInfoHeader.Version == 2)
		{
			IntelDeviceInfoV2 intelDeviceInfo;
			memcpy(&intelDeviceInfo, intelDeviceInfoBuffer, intelDeviceInfoHeader.Size);
			if (intelDeviceInfo.GTGeneration >= IGFX_SKYLAKE)
				return true;
		}

	}
	return false;
}

/*****************************************************************************************
* main
*
*     Function represents the game or application. The application checks for graphics
*     capabilities here and makes whatever decisions it needs to based on the results.
*
*****************************************************************************************/
int main( int argc, char* argv[] )
{
	//
	// First we need to check if it is an Intel processor. If not then an alternate
	// path should be taken. This is also a good place to check for other CPU capabilites
	// such as AVX support.
	//
	std::string CPUBrand;
	std::string CPUVendor;

	getCPUInfo(&CPUBrand, &CPUVendor);
	if (CPUVendor != "GenuineIntel") {
		printf("Not an Intel CPU");
		_getch();
		return 0;
	}

	//
	// The brand string can be parsed to discover some information about the type of
	// processor.
	//
	printf( "CPU Brand: %s\n", CPUBrand.c_str() );

	std::size_t found = CPUBrand.find("i7");
	if (found != std::string::npos)
	{
		printf( "i7 Brand Found\n" );
	}

	found = CPUBrand.find("i5");
	if (found != std::string::npos)
	{
		printf( "i5 Brand Found\n" );
	}

	found = CPUBrand.find("i3");
	if (found != std::string::npos)
	{
		printf( "i3 Brand Found\n" );
	}

	found = CPUBrand.find("Celeron");
	if (found != std::string::npos)
	{
		printf( "Celeron Brand Found\n" );
	}

	found = CPUBrand.find("Pentium");
	if (found != std::string::npos)
	{
		printf( "Pentium Brand Found\n" );
	}

	if (IsGPUOK())
		printf("GPU ok\n");
	else
		printf("GPU not support\n");

    printf("\n");
    printf("Press any key to exit...\n");

	_getch();
    return 0;
}

/*****************************************************************************************
* getDefaultFidelityPresets
*
*     Function to find the default fidelity preset level, based on the type of
*     graphics adapter present.
*
*     The guidelines for graphics preset levels for Intel devices is a generic one
*     based on our observations with various contemporary games. You would have to
*     change it if your game already plays well on the older hardware even at high
*     settings.
*
*****************************************************************************************/

PresetLevel getDefaultFidelityPresets(unsigned int VendorId, unsigned int DeviceId)
{
	//
	// Look for a config file that qualifies devices from any vendor
	// The code here looks for a file with one line per recognized graphics
	// device in the following format:
	//
	// VendorIDHex, DeviceIDHex, CapabilityEnum      ;Commented name of card
	//

	FILE *fp = NULL;
	const char *cfgFileName;

	switch (VendorId)
	{
	case 0x8086:
		cfgFileName = "IntelGfx.cfg";
		break;
		//case 0x1002:
		//    cfgFileName =  "ATI.cfg"; // not provided
		//    break;

		//case 0x10DE:
		//    cfgFileName = "Nvidia.cfg"; // not provided
		//    break;

	default:
		break;
	}

	fopen_s(&fp, cfgFileName, "r");

	PresetLevel presets = Undefined;

	if (fp)
	{
		char line[100];
		char* context = NULL;

		char* szVendorId = NULL;
		char* szDeviceId = NULL;
		char* szPresetLevel = NULL;

		//
		// read one line at a time till EOF
		//
		while (fgets(line, 100, fp))
		{
			//
			// Parse and remove the comment part of any line
			//
			int i; for (i = 0; line[i] && line[i] != ';'; i++); line[i] = '\0';

			//
			// Try to extract VendorId, DeviceId and recommended Default Preset Level
			//
			szVendorId = strtok_s(line, ",\n", &context);
			szDeviceId = strtok_s(NULL, ",\n", &context);
			szPresetLevel = strtok_s(NULL, ",\n", &context);

			if ((szVendorId == NULL) ||
				(szDeviceId == NULL) ||
				(szPresetLevel == NULL))
			{
				continue;  // blank or improper line in cfg file - skip to next line
			}

			unsigned int vId, dId;
			sscanf_s(szVendorId, "%x", &vId);
			sscanf_s(szDeviceId, "%x", &dId);

			//
			// If current graphics device is found in the cfg file, use the 
			// pre-configured default Graphics Presets setting.
			//
			if ((vId == VendorId) && (dId == DeviceId))
			{
				char s[10];
				sscanf_s(szPresetLevel, "%s", s, _countof(s));

				if (!_stricmp(s, "Low"))
					presets = Low;
				else if (!_stricmp(s, "Medium"))
					presets = Medium;
				else if (!_stricmp(s, "Medium+"))
					presets = MediumPlus;
				else if (!_stricmp(s, "High"))
					presets = High;
				else
					presets = NotCompatible;

				break;
			}
		}

		fclose(fp);  // Close open file handle
	}
	else
	{
		printf("%s not found! Presets undefined.\n", cfgFileName);
	}

	//
	// If the current graphics device was not listed in any of the config
	// files, or if config file not found, use Low settings as default.
	// This should be changed to reflect the desired behavior for unknown
	// graphics devices.
	//
	if (presets == Undefined) {
		presets = Low;
	}

	return presets;
}
