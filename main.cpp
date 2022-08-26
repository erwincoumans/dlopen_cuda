#include <iostream>
#include <dlfcn.h>
using namespace std;

#define DYNAMIC_CUDA_PATH "/usr/lib/x86_64-linux-gnu/libcuda.so"
#define DYNAMIC_CUDART_PATH "/usr/local/cuda/lib64/libcudart.so"

enum TINY_CUDA_CODES
{
  CUDA_SUCCESS                              = 0,
  CU_GET_PROC_ADDRESS_DEFAULT = 0,
  cudaEnableDefault = 0,
};

// CUDA driver API functions.
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct cudaArray_st* cudaArray_t;

// see https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html#driver-vs-runtime-api
// cuda driver (cuda.so)
TINY_CUDA_CODES (*cuDriverGetVersion)(int *version);
TINY_CUDA_CODES (*cuInit)(unsigned int flags);
TINY_CUDA_CODES (*cuDeviceGetCount)(int *count);
TINY_CUDA_CODES (*cuDeviceGetCount2)(int *count);
TINY_CUDA_CODES (*cuGetProcAddress) ( const char* symbol, void** pfn, int  cudaVersion, uint64_t flags );
TINY_CUDA_CODES (*cudaMemcpyFromArray)  ( void* dst, const cudaArray_t src, size_t wOffset, size_t hOffset, size_t count, unsigned int kind );

#define LOAD_CUDA_FUNCTION(name, version) \
  name = reinterpret_cast<decltype(name)>(dlsym(cuda_lib , #name version)); \
  if (!name) cout << "Error:" << #name <<  " not found in CUDA library" << endl

// cuda runtime (cudart.so)
TINY_CUDA_CODES (*cudaGraphicsGLRegisterImage)( cudaGraphicsResource** resource, uint64_t image, int target, unsigned int  flags );
//Returns the requested driver API function pointer.
//see also https://forums.developer.nvidia.com/t/some-questions-about-cugetprocaddress/191749
TINY_CUDA_CODES (*cudaGetDriverEntryPoint) ( const char* symbol, void** funcPtr, unsigned long long flags );
TINY_CUDA_CODES (*cudaGraphicsMapResources) (	int 	count, cudaGraphicsResource* resources, cudaStream_t stream );
TINY_CUDA_CODES (*cudaGraphicsSubResourceGetMappedArray) ( cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int  arrayIndex, unsigned int  mipLevel );
TINY_CUDA_CODES (*cudaGraphicsUnmapResources) (	int 	count, cudaGraphicsResource_t * 	resources, cudaStream_t 	stream );

#define LOAD_CUDART_FUNCTION(name, version) \
  name = reinterpret_cast<decltype(name)>(dlsym(cudart_lib , #name version)); \
  if (!name) cout << "Error:" << #name <<  " not found in CUDA library" << endl

int main(int argc, char* argv[])
{
    const char* cuda_lib_path = DYNAMIC_CUDA_PATH;
    void* cuda_lib = dlopen(cuda_lib_path, RTLD_NOW);
    if (!cuda_lib) {
        cout << "Unable to load library " << cuda_lib_path << endl << dlerror() << endl;
        return false;
    }

    void* cudart_lib = dlopen(DYNAMIC_CUDART_PATH, RTLD_NOW);
    if (!cudart_lib) {
        cout << "Unable to load library " << DYNAMIC_CUDART_PATH << endl << dlerror() << endl;
        return false;
    }

    cout << "hello cuda world" << endl;

    LOAD_CUDA_FUNCTION(cuDriverGetVersion, "");
    LOAD_CUDA_FUNCTION(cuInit, "");
    LOAD_CUDA_FUNCTION(cuDeviceGetCount, "");
    LOAD_CUDA_FUNCTION(cuGetProcAddress, "");
        
    LOAD_CUDART_FUNCTION(cudaGetDriverEntryPoint, "");
    LOAD_CUDART_FUNCTION(cudaGraphicsGLRegisterImage, "");
    LOAD_CUDART_FUNCTION(cudaGraphicsMapResources, "");
    LOAD_CUDART_FUNCTION(cudaGraphicsSubResourceGetMappedArray, "");
    LOAD_CUDART_FUNCTION(cudaMemcpyFromArray, "");
    LOAD_CUDART_FUNCTION(cudaGraphicsUnmapResources, "");
    
    auto result = cuInit(0);
    int cuda_driver_version;
    result = cuDriverGetVersion(&cuda_driver_version);
    cout << "CUDA driver version:" << cuda_driver_version << endl;
    int device_count=0;
    result = cuDeviceGetCount(&device_count);
    cout << "CUDA device count:" << device_count << endl;


    result = cuGetProcAddress("cuDeviceGetCount", (void**)&cuDeviceGetCount2, cuda_driver_version, CU_GET_PROC_ADDRESS_DEFAULT);
    if (CUDA_SUCCESS != result)
    {
        cout << "cuDeviceGetCount not found" << endl;
    }
}
