#include "cudahook.h"

#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <chrono>
#include <thread>
#include <algorithm>


#define DEBUG 0

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
	devices().push_back(deviceInfo());

	return ret;
}

typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t *pStream);
static cudaStreamCreate_t realCudaStreamCreate = NULL;

extern "C" cudaError_t cudaStreamCreate(cudaStream_t *pStream) {

	if (realCudaStreamCreate == NULL)
		realCudaStreamCreate = (cudaStreamCreate_t) dlsym(RTLD_NEXT, "cudaStreamCreate");

	assert(realCudaStreamCreate != NULL && "cudaStreamCreate is null");

	return realCudaStreamCreate(pStream);
}

typedef cudaError_t (*cudaFree_t)(void *devPtr);
static cudaFree_t realCudaFree = NULL;

extern "C" cudaError_t cudaFree(void *devPtr) {

	if (realCudaFree == NULL)
		realCudaFree = (cudaFree_t) dlsym(RTLD_NEXT, "cudaFree");

	assert(realCudaFree != NULL && "cudaFree is null");

	return realCudaFree(devPtr);
}

void printDevices() {
	for(auto d : devices()) {
		printf("##################################################\n");
		printf("numOfSMs=%s\n", d.numOfSMs);
		printf("numOfRegister=%s\n", d.numOfRegister);
		printf("sharedMemory=%s\n", d.sharedMemory);
		printf("maxThreads=%s\n", d.maxThreads);
		printf("##################################################\n");
	}
}

/*void printKernels() {
	for(auto k : kernels()) {
		printf("##################################################\n");
		//printf("entry=%d\n", k.entry);
		printf("numOfBlocks=%d\n", k.numOfBlocks);
		printf("numOfThreads=%d\n", k.numOfThreads);
		printf("numOfRegisters=%d\n", k.numOfRegisters);
		printf("sharedMemory=%d\n", k.sharedDynamicMemory);
		printf("sharedMemory=%d\n", k.sharedStaticMemory);
		//printf("computationalTime=%d\n", k.computationalTime);
		printf("##################################################\n");
	}
}*/


std::condition_variable cvm;
std::mutex cv_m;

std::condition_variable cvx;
std::mutex cv_x;
//void knapsack(std::vector<std::vector<int>>& tab, int itens, int pesoTotal){
void knapsack(int ** tab, SharedMap* kernels, int itens, int pesoTotal){

	int item = 1;
	for(SharedMap::iterator iter = kernels->begin(); iter != kernels->end(); iter++)
	{
		for(int peso = 1; peso <= pesoTotal; peso++) {
			int pesoi = iter->second.numOfThreads;
			if(pesoi <= peso) {
				if(pesoi + tab[item-1][peso-pesoi] > tab[item-1][peso])
					tab[item][peso] = pesoi + tab[item-1][peso-pesoi];
				else
					tab[item][peso] = tab[item-1][peso];
			}
			else {
				tab[item][peso] = tab[item-1][peso];
			}
		}
	}
}

//void fill(std::vector<std::vector<int>>& tab, int itens, int pesoTotal, std::vector<int>& resp){
void fill(int **tab, SharedMap* kernels, int itens, int pesoTotal, std::vector<int>& resp){

	SharedMap::iterator iter = kernels->end();
	iter--;

	// se já calculamos esse estado da dp, retornamos o resultado salvo
	while(itens > 0 && pesoTotal > 0) {
		if(tab[itens][pesoTotal] != tab[itens-1][pesoTotal])
		{
			pesoTotal = pesoTotal - iter->second.numOfThreads;
			//printf("iter->first=%d\n", iter->first);
			resp.push_back(iter->first);
		}
		iter--;
		itens--;
	}

}

void schedule(SharedMap* kernels, std::vector<int>& resp) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int peso = prop.maxThreadsPerMultiProcessor;
	int itens = kernels->size();

	int **tab = new int*[itens+1];
	for(int i = 0; i <= itens; i++) {
		tab[i] = new int[peso+1];
	}

	for(int i = 0; i <= itens; i++) {
		tab[i][0] = 0;
	}

	for(int j = 0; j <= peso; j++) {
		tab[0][j] = 0;
	}

	knapsack(tab, kernels, itens, peso);

	fill(tab, kernels, itens, peso, resp);

	for(int j = 0; j <= itens; j++) {
		delete[] tab[j];
	}
	delete[] tab;
}

extern "C" bool scheduleKernels(int n, int num_streams) {
	bip::managed_shared_memory segment(bip::open_only, "shared_memory");
	SharedMap* kernels = segment.find<SharedMap>("Kernels").first;
	{
		//std::unique_lock<std::mutex> lkg(cv_x);
		while (kernels->size() != n);
	}

	cudaStream_t* streams = new cudaStream_t[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	while(kernels->size() != 0) {
		printf("kernels->size()=%d\n", kernels->size());
		std::vector<int> resp;
		//printf("kernels->size()=%d\n", kernels->size());
		schedule(kernels, resp);
		//printf("kernels->size()=%d\n", kernels->size());

		{
			//std::lock_guard<std::mutex> lk(cv_m);
			int s = 0;
			for(int i : resp) {
				printf("i=%d\n", i);
				kernels->at(i).stream = streams[s];
				kernels->at(i).start = true;
				s = (s+1) % num_streams;
			}
		}
		//cvm.notify_all();

		{
			//std::unique_lock<std::mutex> lk(cv_m);
			for(int i : resp) {
				while(!kernels->at(i).finished);
				kernels->erase(i);
				printf("erased=%d\n", i);
			}
		}


		printf("Waiting 2 seconds...\n");
		std::this_thread::sleep_for(std::chrono::seconds(4));
		printf("kernels->size()=%d\n", kernels->size());
		printf("2 seconds finished...\n");
	}
	cudaFree(streams);
	return true;
}

typedef cudaError_t (*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0) {
	if(DEBUG)
		printf("TESTE 1\n");

	printf("kernelInfo()=%d\n", kernelInfo().id+1);
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, (void*) kernelInfo().entry);

	bip::managed_shared_memory segment(bip::open_only, "shared_memory");
	SharedMap* kernels = segment.find<SharedMap>("Kernels").first;

	kernelInfo_t k;
	k.sharedDynamicMemory = sharedMem;
	k.numOfThreads = blockDim.x * blockDim.y * blockDim.z;
	k.numOfBlocks = gridDim.x * gridDim.y * gridDim.z;
	k.numOfRegisters = attr.numRegs;
	k.sharedStaticMemory = attr.sharedSizeBytes;
	k.start = false;
	k.finished = false;

	{
		//std::lock_guard<std::mutex> lkg(cv_x);
		int* index = segment.find<int>("index").first;
		kernelInfo().id = k.id = *index = (*index) + 1;
		kernels->insert(std::pair<const int, kernelInfo_t>(k.id, k));
		printf("vec-size=%d\n", k.id);
	}
	//cvx.notify_all();

	printf("Waiting... \n");
	while(kernels->at(k.id).start != true);
	//cudaStream_t s = kernels->at(k.id).stream;
	printf("%d...finished waiting.\n", k.id);
	kernels->at(k.id).finished = true;

	std::this_thread::sleep_for(std::chrono::seconds(2));
	if (realCudaConfigureCall == NULL)
		realCudaConfigureCall = (cudaConfigureCall_t) dlsym(RTLD_NEXT, "cudaConfigureCall");

	assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
	return realCudaConfigureCall(gridDim, blockDim, sharedMem, stream);

}

typedef cudaError_t (*cudaLaunch_t)(const char *);
static cudaLaunch_t realCudaLaunch = NULL;

extern "C" cudaError_t cudaLaunch(const char *entry) {

	printf("kernelInfoId=%d\n", kernelInfo().id);
	//std::this_thread::sleep_for(std::chrono::seconds(2));


	if (realCudaLaunch == NULL) {
		realCudaLaunch = (cudaLaunch_t) dlsym(RTLD_NEXT, "cudaLaunch");
	}
	assert(realCudaLaunch != NULL && "cudaLaunch is null");

	return realCudaLaunch(entry);
	//return (cudaError_t)0; //success == 0
}

typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
		const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
		const char *hostFun, char *deviceFun, const char *deviceName,
		int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
		int *wSize) {

	kernelInfo().entry = hostFun;

	if(DEBUG)
		printf("TESTE 0\n");

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
	if(DEBUG)
		printf("TESTE 2\n");

	//kernelInfo().args.push_back(const_cast<void *>(arg));
	if (realCudaSetupArgument == NULL) {
		realCudaSetupArgument = (cudaSetupArgument_t) dlsym(RTLD_NEXT,
				"cudaSetupArgument");
	}
	assert(realCudaSetupArgument != NULL);
	return realCudaSetupArgument(arg, size, offset);
}
