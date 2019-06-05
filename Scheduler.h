/*
 * Scheduler.h
 *
 *  Created on: 23/05/2019
 *      Author: rafael
 */

//#include <cuda_profiler_api.h>
#include <vector>
#include <iostream>

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

class Scheduler {
	std::vector<const char*> *programs;

public:

	Scheduler() {
		programs = new std::vector<const char*>();
		//count = 0;
		/*cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);*/
	}

	static

	void init(int numOfDevices);
	/*void add(std::string str) {
		programs.push_back(str);
	}*/

	static void schedule(const char *entry){
		//Scheduler::count++;
		//printf("count=%d\n", Scheduler::count);
		//programs->push_back(entry);
			//return 8;
		//printf("count=%d\n", programs->size());
		//std::chrono::milliseconds timespan(10000); // or whatever
		//std::this_thread::sleep_for(timespan);
	}

	void execute(){
		//cudaError_t e = cudaLaunch(NULL);
		//executeKernels();
		//printf("count=%d\n", programs->size());
	}
};

#endif /* SCHEDULER_H_ */
