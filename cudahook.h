#include <stdio.h>
#include <dlfcn.h>
#include <cassert>
#include <list>
#include <cuda.h>
#include <vector_types.h>
#include <vector>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

namespace bip = boost::interprocess;

typedef struct {
	const char* entry;
	int id;
	//dim3 gridDim;
	//dim3 blockDim;
	int numOfBlocks;
	int numOfThreads;
	int numOfRegisters;
	int sharedDynamicMemory;
	int sharedStaticMemory;
	int computationalTime;
	cudaStream_t stream;
	bool start = false;
	std::list<void *> args;
} kernelInfo_t;

kernelInfo_t &kernelInfo() {
	static kernelInfo_t _kernelInfo;
	return _kernelInfo;
}

std::vector<kernelInfo_t> &kernels() {
	static std::vector<kernelInfo_t> _kernels;
	return _kernels;
}

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

struct find_kernel
{
    int id;
    find_kernel(int id) : id(id) {}
    bool operator () ( const kernelInfo_t& m ) const
    {
        return m.id == id;
    }
    //it = std::find_if( monsters.begin(), monsters.end(), find_monster(monsterID));
};

typedef bip::allocator<kernelInfo_t, boost::interprocess::managed_shared_memory::segment_manager> ShmemListAllocator;
typedef bip::vector<kernelInfo_t, ShmemListAllocator> MyVector;
