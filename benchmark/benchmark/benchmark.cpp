// benchmark.cpp : 定义控制台应用程序的入口点。
//

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <ctime>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "Configure.h"
#include "InferPerf.h"

using std::string;

#define CONFIGFILE	"configure.txt"

vector<string> split(string strtem, char a)
{
	vector<string> strvec;

	string::size_type pos1, pos2;
	strtem = strtem + a;
	pos2 = strtem.find(a);
	pos1 = 0;
	while (string::npos != pos2)
	{
		if(pos2>pos1){
			string tmp = strtem.substr(pos1, pos2 - pos1);
			string tmp2="";
			for(auto c:tmp){
				if (c!=a && c!='\r' && c!='\n')
					tmp2+=c;
			}
			strvec.push_back(tmp2);
		}
		pos1 = pos2 + 1;
		while(strtem[pos1]==a||strtem[pos1]=='\r'||strtem[pos1]=='\n'){
			pos1++;
		}
		pos2 = strtem.find(a, pos1);
	}

	return strvec;
}

string GetFileVariableValue(string sKey, string sConfigFile)
{
	string sLine, sLine2;
	string sValue;
	size_t found;

	std::ifstream configFile;
	configFile.open(sConfigFile);

	while (!configFile.eof())
	{
		std::getline(configFile, sLine);

		found = sLine.find(sKey);

		if ((found != string::npos) && !sLine.find_first_not_of('#'))
		{
			sValue = sLine.substr(sLine.find_first_of("=") + 1);
		}
	}
	configFile.close();

	if (sValue.empty())
	{
		printf("***WARNING: %s is empty.\n", sKey.c_str());
	}
	return sValue;
}

string GenerateWeights(const string& path, const string& xmlname)
{
	std::ifstream xmlfile;
	string sLine;
	size_t found, found2, last;
	std::streamsize maxoffset = -1;
	std::streamsize maxsize = 0;

	xmlfile.open(path+ xmlname+".xml");
	if (!xmlfile.is_open())
		return "XMLNA:"+ path + xmlname + ".xml";
	while (!xmlfile.eof())
	{
		std::getline(xmlfile, sLine);

		found = sLine.find("offset=");
		found2 = sLine.find("size=");
		last = sLine.find_last_of('"');

		if ((found != string::npos) && (found2 != string::npos) && (last>found2+6))
		{
			if (found2 > found + 9) {
				int offset = std::stoi(sLine.substr(found + 8, found2 - found - 9));
				if (offset > maxoffset) {
					maxoffset = offset;
					maxsize = std::stoi(sLine.substr(found2 + 6, last - found2 - 6));
				}
			}
		}
	}
	xmlfile.close();

	if (maxoffset > 0 && maxsize > 0)
	{
		std::streamsize total = maxoffset + maxsize;
		std::ofstream binfile(path + xmlname + ".bin", std::ios::binary);
		if (binfile.is_open()) {
			char buf[8192] = { 0 };
			while (total > 0) {
				binfile.write(buf, total > 8192 ? 8192 : total);
				total -= 8192;
			}
			return "";
		}
		else
			return "weight file create fail";
	}
	return "unknow error";
}

#define STROUT(a) resultfile a;std::cout a;
int main()
{
#ifdef _WIN32
	_mkdir("result");
#else
	mkdir("result", 0777);
#endif
	string spacestring = "                                                 ";
	time_t t = time(0);
	char ch[64];
	strftime(ch, sizeof(ch), "%Y%m%d%H%M%S", localtime(&t)); //年-月-日 时-分-秒

	std::ofstream resultfile("./result/benchmark_"+ string(ch)+".txt");
	STROUT( << "-----------------------------------------");
	strftime(ch, sizeof(ch), "%Y-%m-%d %H:%M:%S", localtime(&t)); //年-月-日 时-分-秒
	STROUT( << ch);
	STROUT(<< "----------------------------------------------" << std::endl);

	Configure::getInstance().devices = split(GetFileVariableValue("INFERDEVICE", CONFIGFILE), ' ');
	Configure::getInstance().precision = split(GetFileVariableValue("MODELPRECISION", CONFIGFILE), ' ');
	Configure::getInstance().modelnames = split(GetFileVariableValue("MODELNAME", CONFIGFILE), ' ');
	vector<string> batchs = split(GetFileVariableValue("BATCHRANGE", CONFIGFILE), ' ');
	for (string s : batchs) {
		try {
			int b = std::stoi(s);
			Configure::getInstance().batchs.push_back(b);
		}
		catch (...) {
			printf("!!!Error: BATCHRANGE must be number.\n");
			return -1;
		}
	}

	//do test
	for (string device : Configure::getInstance().devices) {
		for (string precision : Configure::getInstance().precision) {
			if (device.find("CPU") != string::npos && precision.find("FP16") != string::npos)
				continue;
			STROUT (<< "Device:"<< device<<"  Precision:" << precision << std::endl);
			STROUT (<< "ModelName        ");  //all 8 spaces
			for (int batch : Configure::getInstance().batchs) {
				if (batch < 10) {
					STROUT(<< batch << "        ");
				}
				else if (batch < 100) {
					STROUT(<< batch << "       ");
				}
				else {
					STROUT(<< batch << "      ");
				}
			}
			STROUT (<< std::endl);

			for (string model : Configure::getInstance().modelnames) {
				STROUT (<< model << spacestring.substr(0, 17 - model.length()));
				string ret = GenerateWeights("./model/" + precision + "/", model);
				if (ret.length() > 0) {
					STROUT (<< ret);
				}
				else {
					for (int batch : Configure::getInstance().batchs) {
						InferPerf perf;
						perf.Load("./model/" + precision + "/", model, device,batch);
						if (perf.err_msg.length() > 0) {
							STROUT (<< perf.err_msg);
							break;
						}
						string retperf = perf.Perf();
						STROUT(<< retperf << spacestring.substr(0, 9 - retperf.length()));
					}
				}
				STROUT (<< std::endl);
				string binfilename = "./model/" + precision + "/" + model + ".bin";
				remove(binfilename.c_str());
			}
			STROUT (<< std::endl);
		}
	}

	resultfile.close();
    return 0;
}



