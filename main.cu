#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
//#include <cuda_profiler_api.h>
#include <iostream>

#include <vector>
#include <thread>
#include <future>
#include "Scheduler.h"
#include "cudahook.h"

#include <unistd.h>
#include <dlfcn.h>
#include <signal.h>

#include <boost/interprocess/managed_shared_memory.hpp>

namespace bip = boost::interprocess;
//namespace bv = boost::container;

typedef void* my_lib_t;

my_lib_t MyLoadLib(const char* szMyLib) {
	return dlopen(szMyLib, RTLD_LAZY);
}

void MyUnloadLib(my_lib_t hMyLib) {
	dlclose(hMyLib);
}

void* MyLoadProc(my_lib_t hMyLib, const char* szMyProc) {
	return dlsym(hMyLib, szMyProc);
}

typedef bool (*scheduleKernels_t)(int, int);
my_lib_t hMyLib = NULL;
scheduleKernels_t scheduleKernels = NULL;

bool callcudahook(int n, int streams) {
  if (!(hMyLib = MyLoadLib("/home/rafael/cuda-workspace/wrappercuda/libcudahook.so"))) { /*error*/ }
  if (!(scheduleKernels = (scheduleKernels_t)MyLoadProc(hMyLib, "scheduleKernels"))) { /*error*/ }

  bool ret = scheduleKernels(n, streams);

  MyUnloadLib(hMyLib);

  return ret;
}


void exec(const char* s){
	system(s);
}

int main(int argc, char **argv) {

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	/*bip::shared_memory_object::remove("shared_memory");
	bip::managed_shared_memory managed_shm(bip::open_or_create, "shared_memory", 1024);
	int *i = managed_shm.construct<int>("index")(0);
	std::cout << *i << '\n';*/

	bip::shared_memory_object::remove("MySharedMemory");
	bip::managed_shared_memory segment(boost::interprocess::create_only, "MySharedMemory", 65536);
	MyVector *kernels = segment.construct<MyVector>("Kernels")(segment.get_segment_manager());

	std::string line1 = "";
	std::string line2 = "";

	std::vector<std::future<void>> vec;
	std::getline (std::cin, line1);
	vec.push_back(std::async(std::launch::async,exec,line1.data()));

	std::getline (std::cin, line2);
	vec.push_back(std::async(std::launch::async,exec,line2.data()));


	callcudahook(2, 2);

	vec[0].get();
	vec[1].get();

	/*std::cout << *i << '\n';
	std::pair<int*, std::size_t> p = managed_shm.find<int>("index");
	if (p.first)
		std::cout << *p.first << '\n';*/


	//boost::interprocess::shared_memory_object::remove("shared_memory");
	//boost::interprocess::shared_memory_object::remove("index");

	return 0;
}


/*std::vector<std::future<void>> vec;

	std::string line1 = "";
	std::string line2 = "";
	std::string line3 = "";
	std::string line4 = "";
	std::string line5 = "";
	std::string line6 = "";
	std::string line7 = "";
	std::string line8 = "";

	std::getline (std::cin, line1);
	std::getline (std::cin, line2);
	std::getline (std::cin, line3);
	std::getline (std::cin, line4);


	std::vector<char*> commandVector;
	commandVector.push_back(const_cast<char*>(line2.data()));
	commandVector.push_back(const_cast<char*>(line3.data()));
	commandVector.push_back(const_cast<char*>(line4.data()));
	commandVector.push_back(NULL);
	//const int status = execvp(commandVector[0], &commandVector[0]);
	//exec(commandVector[0], commandVector);
	myclass a(commandVector[0], commandVector);

	std::vector<char*> commandVector2;
	//commandVector2.push_back(const_cast<char*>(line1.data()));
	std::getline (std::cin, line1);
	std::getline (std::cin, line2);
	std::getline (std::cin, line3);
	std::getline (std::cin, line4);
	std::getline (std::cin, line5);
	std::getline (std::cin, line6);
	std::getline (std::cin, line7);
	std::getline (std::cin, line8);

	commandVector2.push_back(const_cast<char*>(line2.data()));
	commandVector2.push_back(const_cast<char*>(line3.data()));
	commandVector2.push_back(const_cast<char*>(line4.data()));
	commandVector2.push_back(const_cast<char*>(line5.data()));
	commandVector2.push_back(const_cast<char*>(line6.data()));
	commandVector2.push_back(const_cast<char*>(line7.data()));
	commandVector2.push_back(const_cast<char*>(line8.data()));
	commandVector2.push_back(NULL);
	//const int status = execvp(commandVector[0], &commandVector[0]);
	//exec(commandVector2[0], commandVector2);
	myclass b(commandVector2[0], commandVector2);*/
