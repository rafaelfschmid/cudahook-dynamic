/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
//#include <cuda_profiler_api.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include "Scheduler.h"

void exec(const char* s){
	system(s);
}
//cudaStream_t streams[NUM_STREAMS];

int main(int argc, char **argv) {

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	std::string line = "";
	std::getline (std::cin, line);
	exec(line.data());
	//s.add(line);
	//std::thread t1(exec,line.data());

	std::string line2 = "";
	std::getline (std::cin, line2);
	//exec(line.data());
	exec(line2.data());
	//std::thread t2(exec,line2.data());

	//rodinia::main(argc, argv);

	//Scheduler s;
	//int *num = Scheduler::num;
	//s.init(0);
	//try{}
	//catch(...){printf("Exception\n");}
	/*
	std::string line = "";
	while(line != " ") {
		std::getline (std::cin, line);
		//std::cout << line << "\n";
		std::string str = argv[i];//"./hotspot 1024 2 2 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.out";
		s.programCall(line);
	//	s.schedule();
	}*/
	//s.execute();

	return 0;
}
