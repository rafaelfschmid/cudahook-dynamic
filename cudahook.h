#include <stdio.h>
#include <dlfcn.h>
#include <cassert>
#include <list>
#include <cuda.h>
#include <vector_types.h>
#include <vector>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/map.hpp>

namespace bip = boost::interprocess;

typedef struct {
	const char* entry;
	int id = -1;
	//dim3 gridDim;
	//dim3 blockDim;
	int numOfBlocks;
	int numOfThreads;
	int numOfRegisters;
	int sharedDynamicMemory;
	int sharedStaticMemory;
	//int computationalTime;
	cudaStream_t stream;
	bool start = false;
	bool finished = false;
	//std::list<void *> args;
} kernelInfo_t;

kernelInfo_t &kernelInfo() {
	static kernelInfo_t _kernelInfo;
	return _kernelInfo;
}

/*std::vector<kernelInfo_t> &kernels() {
	static std::vector<kernelInfo_t> _kernels;
	return _kernels;
}*/

typedef struct {
	int numOfSMs;
	int numOfRegister; // register per SM
	int maxThreads;    // max threads per SM
	int sharedMemory;  // sharedMemory per SM
} deviceInfo_t;

deviceInfo_t &deviceInfo() {
	static deviceInfo_t _deviceInfo;
	return _deviceInfo;
}

std::vector<deviceInfo_t> &devices() {
	static std::vector<deviceInfo_t> _devices;
	return _devices;
}

typedef int    KeyType;
typedef kernelInfo_t  MappedType;
typedef std::pair<const int, kernelInfo_t> ValueType;

//allocator of for the map.
typedef bip::allocator<ValueType, bip::managed_shared_memory::segment_manager> ShmemAllocator;

//third parameter argument is the ordering function is used to compare the keys.
typedef bip::map<int, kernelInfo_t, std::less<int>, ShmemAllocator> SharedMap;
