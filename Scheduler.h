/*
 * Scheduler.h
 *
 *  Created on: 23/05/2019
 *      Author: rafael
 */

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

class Scheduler {
public:
	Scheduler() {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
	}

	void init(int numOfDevices)
	{
		int count;
		cudaGetDeviceCount(&count);

		for(int i = 0; i < count; i++){
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, i);
		}
	}

	void schedule(){

	}

	void execute(){
		doLau
	}
};

#endif /* SCHEDULER_H_ */
