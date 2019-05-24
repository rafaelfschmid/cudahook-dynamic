#include <stdio.h>
#include <dlfcn.h>
#include <cassert>
#include <map>
#include <list>
#include<vector>

#include <cuda.h>
#include <vector_types.h>

//#include "cudahook.h"

typedef struct {
	const char* entry;
	char* deviceFun;
	dim3  gridDim;
	dim3  blockDim;
	int   counter;
	std::list<void *> args;
} kernelInfo_t;

/*typedef struct {
 int numOfblocks;
 int numOfThreads;
 int numOfRegisters;
 int sharedMemory;
 int computationalTime;
 std::list<void *> args;
 } kernelInfo_t;*/

kernelInfo_t &kernelInfo() {
	static kernelInfo_t _kernelInfo;
	return _kernelInfo;
}

std::vector<kernelInfo_t> &kernels() {
	static std::vector<kernelInfo_t> _kernels;
	return _kernels;
}

/*std::map<const char *, char *> &kernels() {
	static std::map<const char*, char*> _kernels;
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

/*std::vector<deviceInfo_t> &devices() {
	static std::vector<deviceInfo_t> _devices;
	return _devices;
}*/

void print_kernel_invocation(const char *entry) {
	dim3 gridDim = kernelInfo().gridDim;
	dim3 blockDim = kernelInfo().blockDim;

	printf("SENTINEL %s ", kernelInfo().deviceFun);
	if (gridDim.y == 1 && gridDim.z == 1) {
		printf("--gridDim=%d ", gridDim.x);
	} else if (gridDim.z == 1) {
		printf("--gridDim=[%d,%d] ", gridDim.x, gridDim.y);
	} else {
		printf("--gridDim=[%d,%d,%d] ", gridDim.x, gridDim.y, gridDim.z);
	}
	if (blockDim.y == 1 && blockDim.z == 1) {
		printf("--blockDim=%d ", blockDim.x);
	} else if (blockDim.z == 1) {
		printf("--blockDim=[%d,%d] ", blockDim.x, blockDim.y);
	} else {
		printf("--blockDim=[%d,%d,%d] ", blockDim.x, blockDim.y, blockDim.z);
	}
	for (std::list<void *>::iterator it = kernelInfo().args.begin(), end =
			kernelInfo().args.end(); it != end; ++it) {
		unsigned i = std::distance(kernelInfo().args.begin(), it);
		printf("%d:%d ", i, *(static_cast<int *>(*it)));
	}
	printf("\n");
}

typedef cudaError_t (*cudaFuncGetAttributes_t)(struct cudaFuncAttributes *,	const void *);
static cudaFuncGetAttributes_t realCudaFuncGetAttributes = NULL;

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {

	if (realCudaFuncGetAttributes == NULL)
		realCudaFuncGetAttributes = (cudaFuncGetAttributes_t) dlsym(RTLD_NEXT,
				"cudaFuncGetAttributes");

	assert(realCudaFuncGetAttributes != NULL && "cudaFuncGetAttributes is null");

	return realCudaFuncGetAttributes(attr, func);
}

typedef cudaError_t (*cudaGetDeviceProperties_t)(struct cudaDeviceProp *prop, int device);
static cudaGetDeviceProperties_t realCudaGetDeviceProperties = NULL;

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,	int device) {

	if (realCudaGetDeviceProperties == NULL)
		realCudaGetDeviceProperties = (cudaGetDeviceProperties_t) dlsym(RTLD_NEXT, "cudaGetDeviceProperties");

	assert(realCudaGetDeviceProperties != NULL && "cudaGetDeviceProperties is null");

	auto ret = realCudaGetDeviceProperties(prop, device);

	deviceInfo().numOfSMs = prop->multiProcessorCount;
	deviceInfo().numOfRegister = prop->regsPerMultiprocessor;
	deviceInfo().sharedMemory = prop->sharedMemPerMultiprocessor;
	deviceInfo().maxThreads = prop->maxThreadsPerMultiProcessor;

	printf("Device name:     %s\n", prop->name);
	printf("num of SMs:      %d\n", prop->multiProcessorCount);
	printf("num of register: %d\n", prop->regsPerMultiprocessor);
	printf("max threads:     %d\n", prop->sharedMemPerMultiprocessor);
	printf("sharedMemory:    %d\n", prop->maxThreadsPerMultiProcessor);

	return ret;
}

typedef cudaError_t (*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0) {
	//printf("TESTE 1\n");
	assert(kernelInfo().counter == 0 && "Multiple cudaConfigureCalls before cudaLaunch?");
	kernelInfo().gridDim = gridDim;
	kernelInfo().blockDim = blockDim;
	kernelInfo().counter++;

	if (realCudaConfigureCall == NULL)
		realCudaConfigureCall = (cudaConfigureCall_t) dlsym(RTLD_NEXT, "cudaConfigureCall");

	assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
	return realCudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

typedef cudaError_t (*cudaLaunch_t)(const char *);
static cudaLaunch_t realCudaLaunch = NULL;

extern "C" cudaError_t cudaLaunch(const char *entry) {

	//printf("TESTE 3\n");

	assert(kernelInfo().counter == 1 && "Multiple cudaConfigureCalls before cudaLaunch?");

	print_kernel_invocation(entry);
	kernelInfo().counter--;
	kernelInfo().args.clear();

	cudaFuncAttributes attr;
	//cudaFuncGetAttributes(&attr, kernels()[entry]);
	cudaFuncGetAttributes(&attr, (void*) entry);
	printf("######################################################\n");
	printf("binaryVersion=%d\n", attr.binaryVersion);
	printf("cacheModeCA=%d\n", attr.cacheModeCA);
	printf("constSizeBytes=%d\n", attr.constSizeBytes);
	printf("localSizeBytes=%d\n", attr.localSizeBytes);
	printf("maxThreadsPerBlock=%d\n", attr.maxThreadsPerBlock);
	printf("numRegs=%d\n", attr.numRegs);
	printf("ptxVersion=%d\n", attr.ptxVersion);
	printf("sharedSizeBytes=%d\n", attr.sharedSizeBytes);
	printf("######################################################\n");

	if (realCudaLaunch == NULL) {
		realCudaLaunch = (cudaLaunch_t) dlsym(RTLD_NEXT, "cudaLaunch");
	}
	assert(realCudaLaunch != NULL && "cudaLaunch is null");

	//return realCudaLaunch(entry);
	return (cudaError_t)0; //success == 0
}

typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
		const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
		const char *hostFun, char *deviceFun, const char *deviceName,
		int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
		int *wSize) {

	//printf("TESTE 0\n");
	kernelInfo().entry = hostFun;
	kernelInfo().deviceFun = deviceFun;
	//kernels()[hostFun] = deviceFun;

	if (realCudaRegisterFunction == NULL) {
		realCudaRegisterFunction = (cudaRegisterFunction_t) dlsym(RTLD_NEXT,
				"__cudaRegisterFunction");
	}
	assert(realCudaRegisterFunction != NULL && "cudaRegisterFunction is null");

	realCudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
			thread_limit, tid, bid, bDim, gDim, wSize);
}

typedef cudaError_t (*cudaSetupArgument_t)(const void *, size_t, size_t);
static cudaSetupArgument_t realCudaSetupArgument = NULL;

extern "C" cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	//printf("TESTE 2\n");

	kernelInfo().args.push_back(const_cast<void *>(arg));
	if (realCudaSetupArgument == NULL) {
		realCudaSetupArgument = (cudaSetupArgument_t) dlsym(RTLD_NEXT,
				"cudaSetupArgument");
	}
	assert(realCudaSetupArgument != NULL);
	return realCudaSetupArgument(arg, size, offset);
}
