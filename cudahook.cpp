#include "cudahook.h"

#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <chrono>
#include <thread>
#include <algorithm>


#define DEBUG 1

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

void printKernels() {
	for(auto k : kernels()) {
		printf("##################################################\n");
		//printf("entry=%d\n", k.entry);
		printf("numOfBlocks=%d\n", k.numOfBlocks);
		printf("numOfThreads=%d\n", k.numOfThreads);
		printf("numOfRegisters=%d\n", k.numOfRegisters);
		printf("sharedMemory=%d\n", k.sharedDynamicMemory);
		printf("sharedMemory=%d\n", k.sharedStaticMemory);
		printf("computationalTime=%d\n", k.computationalTime);
		printf("##################################################\n");
	}
}

std::condition_variable cvm;
std::mutex cv_m;

std::condition_variable cvx;
std::mutex cv_x;

/*struct Comp {
	template<typename T>
	bool operator()(const T& t1, const T& t2) const
	{
		return t1.numOfRegisters < t2.numOfRegisters;
	}
};*/

void knapsack(int **tab, int itens, int pesoTotal){

	for(int item = 1; item <= itens; item++) {
		for(int peso = 1; peso <= pesoTotal; peso++) {
			if(kernels()[item-1].start) {
				tab[item][peso] = tab[item-1][peso];
			}
			else {
				int pesoi = kernels()[item-1].numOfThreads;
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
}

void fill(int **tab, int itens, int pesoTotal, int* resp){

	// se já calculamos esse estado da dp, retornamos o resultado salvo
	while(itens > 0 && pesoTotal > 0) {
		if(tab[itens][pesoTotal] != tab[itens-1][pesoTotal])
		{
			resp[itens] = 1;
			pesoTotal = pesoTotal - kernels()[itens-1].numOfThreads;
		}
		itens--;
	}
}


extern "C" bool scheduleKernels(int n, int num_streams) {
	/*{
		std::unique_lock<std::mutex> lkg(cv_x);
		cvx.wait(lkg, [&n](){ return kernels().size() == n; });
		if(DEBUG)
			printf("kernels().size()=%d == n=%d\n", kernels().size(), n);
	}*/

	//bip::managed_shared_memory managed_shm(bip::open_only, "shared_memory");
	{
		std::unique_lock<std::mutex> lkg(cv_x);
		//std::this_thread::sleep_for(std::chrono::seconds(1));
		//std::pair<int*, std::size_t> p = managed_shm.find<int>("index");
		//bip::managed_shared_memory segment(bip::open_only, "MySharedMemory");
		bip::managed_shared_memory segment(bip::open_only, "MySharedMemory");
		MyVector* myVector = segment.find<MyVector>("Kernels").first;
		//cvx.wait(lkg, [&n, &p](){return (*p.first) == n;});
		//printf("*p.first=%d -------- n=%d\n", *p.first, n);
		//while (*p.first != n);
		while (myVector->size() != n);
    	//printf("*p.first=%d -------- n=%d\n", *p.first, n);

		//
	}

	/*int peso = devices()[0].maxThreads;
	int itens = kernels().size();
	int count = 0;
	int s = 0;

	printKernels();

	cudaStream_t* streams = new cudaStream_t[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	int **tab = new int*[itens+1];
	int *resp = new int[itens+1];
	for(int i = 0; i <= itens; i++) {
		tab[i] = new int[peso+1];
	}

	while(count != itens) {
		/*for(int i = 0; i <= itens; i++) {
			tab[i][0] = 0;
			resp[i] = 0;
		}

		for(int j = 0; j <= peso; j++) {
			tab[0][j] = 0;
		}

		for(int i = 0; i <= itens; i++) {
			resp[i] = 0;
			for(int j = 0; j <= peso; j++) {
				tab[i][j] = 0;
			}
		}


		knapsack(tab, itens, peso);
		fill(tab, itens, peso, resp);
		//printf("peso=%d", peso);
		for(int j = 0; j <= itens; j++) {
			printf("%d ",resp[j]);
		}

		//int i = 0;
		//while(i <= itens) {

		for(int i = 1; i <= itens; i++) {
			if(resp[i] == 1) {
				count++;
				//printf("mochila-->i=%d\n", i-1);
				std::lock_guard<std::mutex> lk(cv_m);
				kernels()[i-1].start = true;
				kernels()[i-1].stream = streams[s];
				s = (s+1) % num_streams;
				//a = !a;
				cvm.notify_all();
			}
			//else i++;
		}
	}*/

	cudaStream_t* streams = new cudaStream_t[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	//printf("kernels().size()=%d == n=%d\n", kernels().size(), n);
	int s = 0;
	std::lock_guard<std::mutex> lk(cv_m);
	bip::managed_shared_memory segment(bip::open_only, "MySharedMemory");
	MyVector* myVector = segment.find<MyVector>("Kernels").first;
	for(MyVector::iterator iter = myVector->begin(); iter < myVector->end(); iter++)
	{
		iter->start = true;
		iter->stream = streams[s];
		s = (s+1) % num_streams;
		/*std::string str("kernel" + std::to_string(i));
		printf("kernel=%s\n", str.data());
		std::pair<kernelInfo_t*, std::size_t> p = managed_shm.find<kernelInfo_t>(str.data());
		p.first->start = true;
		p.first->stream = streams[s];
		s = (s+1) % num_streams;*/


	}
	cvm.notify_all();
	/*for(int i = 0; i < kernels().size(); i++)
	{
		printf("id=%d\n", i);
		kernels()[i].start = true;
		kernels()[i].stream = streams[s];
		s = (s+1) % num_streams;
		cvm.notify_all();
	}*/

	/*for(int i = 0; i <= itens; i++) {
		delete tab[i];
	}
	delete resp;
	cudaFree(streams);*/

	return true;
}

typedef cudaError_t (*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0) {
	if(DEBUG)
		printf("TESTE 1\n");

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, (void*) kernelInfo().entry);
	kernelInfo_t k;
	k.sharedDynamicMemory = sharedMem;
	k.numOfThreads = blockDim.x * blockDim.y * blockDim.z;
	k.numOfBlocks = gridDim.x * gridDim.y * gridDim.z;
	k.numOfRegisters = attr.numRegs;
	k.sharedStaticMemory = attr.sharedSizeBytes;

	//bip::managed_shared_memory managed_shm(bip::open_only, "shared_memory");

	bip::managed_shared_memory segment(bip::open_only, "MySharedMemory");
	std::string str;
	int id;
	{
		/*std::lock_guard<std::mutex> lkg(cv_x);
		std::pair<int*, std::size_t> p = managed_shm.find<int>("index");
		id = *p.first;
		*p.first = (*p.first)+1;
		printf("id=%d\n",id);
		str = "kernel" + std::to_string(id);*/
		//str = "kernel"+ std::string(id);
		//str +=id;
		/*printf("kernel=%s\n", str.data());
		k = managed_shm.construct<kernelInfo_t>(str.data())();
		k->sharedDynamicMemory = sharedMem;
		k->numOfThreads = blockDim.x * blockDim.y * blockDim.z;
		k->numOfBlocks = gridDim.x * gridDim.y * gridDim.z;
		k->numOfRegisters = attr.numRegs;
		k->sharedStaticMemory = attr.sharedSizeBytes;*/
		std::lock_guard<std::mutex> lkg(cv_x);
		//bip::managed_shared_memory segment(open_only, "MySharedMemory");
		MyVector* myVector = segment.find<MyVector>("Kernels").first;
		id = k.id = myVector->size();
		myVector->push_back(k);
	}
	cvx.notify_all();

	{
		std::unique_lock<std::mutex> lk(cv_m);
		printf("Waiting... \n");
		//cvm.wait(lk, [&id](){return id == 1;});
		//std::pair<kernelInfo_t*, std::size_t> p = managed_shm.find<kernelInfo_t>(str.data());
		//cvm.wait(lk, [&p](){return p.first->start == true;});
		MyVector* myVector = segment.find<MyVector>("Kernels").first;
		MyVector::iterator iter = myVector->begin();
		iter+=id;
		while(iter->start != true);
		printf("%d...finished waiting.\n", id);
	}

	if (realCudaConfigureCall == NULL)
		realCudaConfigureCall = (cudaConfigureCall_t) dlsym(RTLD_NEXT, "cudaConfigureCall");

	assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
	return realCudaConfigureCall(gridDim, blockDim, sharedMem, kernelInfo().stream);
}

typedef cudaError_t (*cudaLaunch_t)(const char *);
static cudaLaunch_t realCudaLaunch = NULL;

extern "C" cudaError_t cudaLaunch(const char *entry) {

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

	kernelInfo().args.push_back(const_cast<void *>(arg));
	if (realCudaSetupArgument == NULL) {
		realCudaSetupArgument = (cudaSetupArgument_t) dlsym(RTLD_NEXT,
				"cudaSetupArgument");
	}
	assert(realCudaSetupArgument != NULL);
	return realCudaSetupArgument(arg, size, offset);
}
