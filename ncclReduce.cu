/*%****************************************************************************80
%  Code: 
%   ncclReduce.cu
%
%  Purpose:
%   Implements sample reduce code using the package NCCL(ncclReduce).
%   Using 'Multiples Devices per Thread'.
%   Implements dot product(scalar product).
%   x = (xo, x1, x2, ..., xn)
%   y = (yo, y1, y2, ..., yn)
%   c = (xo . yo + x1 . y1 + ..., xn . yn)
%
%  Modified:
%   Aug 18 2020 10:57 
%
%  Author:
%   Murilo Boratto <murilo.boratto 'at' fieb.org.br>
%
%  How to Compile:
%   nvcc ncclReduce.cu -o ncclReduce -lnccl  
%
%  Execute: 
%   ./ncclReduce                           
%   
%****************************************************************************80*/
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "/home/rogerlee/code/nccl/build/include/nccl.h"
#include <math.h>
const int ngpu = 8;


void NCCLInit(ncclComm_t* &comm, cudaStream_t* &s, int* &device){
  comm = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * ngpu);  
  s    = (cudaStream_t*)malloc(sizeof(cudaStream_t)* ngpu);
  device = (int *)malloc(ngpu*sizeof(int));

  for(int g=0;g<ngpu;g++){
    device[g] = g;
    cudaSetDevice(device[g]); 
    cudaStreamCreate(&s[g]);
    cudaDeviceSynchronize();  
    cudaStreamSynchronize(s[g]);
  }
  ncclCommInitAll(comm, ngpu, device);

}

template<class T>
void CudaMemInit(T** &x, const long long data_size, int* device){
  x  = (T**)malloc(ngpu*sizeof(T*));
  for(int g=0;g<ngpu;g++){
    cudaSetDevice(device[g]);
    cudaMalloc(&x[g], data_size * sizeof(T));
  }
}

void ncclfinish(ncclComm_t* &comm, cudaStream_t* &s, int* device){
    for(int g = 0; g < ngpu; g++) { /*Destroy CUDA Streams*/
        cudaSetDevice(device[g]);
        cudaStreamDestroy(s[g]);
    }

    for(int g = 0; g < ngpu; g++) /*Finalizing NCCL*/
      ncclCommDestroy(comm[g]);
}

template<class T>
float allreduceTest(long long data_size, int* device, int loop, ncclComm_t* comm, cudaStream_t* s, T** &src, T** &dst, size_t mode){
  int cnt = 0;
  float elapsedTime = 0.0, totaltime = 0.0;
  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);

  // cudaEvent_t e_start[ngpu];
  // cudaEvent_t e_stop[ngpu];
  // for(int g=0;g<ngpu;g++){
  //   if(cudaEventCreate(&e_start[g])!=cudaSuccess){
  //     printf("Create %d start event ERROR\n",g);
  //     return;
  //   }
  //   if(cudaEventCreate(&e_stop[g])!=cudaSuccess){
  //     printf("Create %d stop event ERROR\n",g);
  //     return;
  //   }      
  // }
    
  
  while(cnt<loop){
    ncclGroupStart(); 
    for(int g = 0; g < ngpu; g++) {
      cudaSetDevice(device[g]);
      // ncclAllReduce(src[g], dst[g], data_size, ncclDouble, ncclSum, comm[g], s[g], mode); 
      // if(split == 0){
      //   ncclAllReduce(src[g], dst[g], data_size, ncclDouble, ncclSum, comm[g], s[g], mode); 
      // }else{
        int offset = data_size/2;
        ncclAllReduce(src[g], dst[g], offset, ncclDouble, ncclSum, comm[g], 0, 3); 
        ncclAllReduce(src[g]+offset, dst[g]+offset, data_size - offset, ncclDouble, ncclSum, comm[g], 0, 2); 
      // }
    }
    ncclGroupEnd();
    ncclGroupStart(); 
    for(int g = 0; g < ngpu; g++) {
      cudaSetDevice(device[g]);
      // ncclAllReduce(src[g], dst[g], data_size, ncclDouble, ncclSum, comm[g], s[g], mode); 
      // if(split == 0){
      //   ncclAllReduce(src[g], dst[g], data_size, ncclDouble, ncclSum, comm[g], s[g], mode); 
      // }else{
        int offset = data_size/2;
        ncclAllReduce(src[g], dst[g], offset, ncclDouble, ncclSum, comm[g], 0, 2); 
        ncclAllReduce(src[g]+offset, dst[g]+offset, data_size - offset, ncclDouble, ncclSum, comm[g], 0, 3); 
      // }
    }
    // for(int g=0;g<ngpu;g++){
    //   cudaEventRecord(e_start[g], s[g]);  
    // }
    cudaEventRecord(e_start, 0);
    ncclGroupEnd();

    for(int g = 0; g < ngpu; g++) {
        cudaSetDevice(device[g]);    
        cudaDeviceSynchronize();  
        cudaStreamSynchronize(s[g]);     
        // cudaEventRecord(e_stop[g], s[g]);   
        // cudaEventSynchronize(e_stop[g]);
        // cudaEventElapsedTime(&totaltime, e_start[g], e_stop[g]);
        // elapsedTime = elapsedTime+totaltime;
    }
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&totaltime, e_start, e_stop);
    elapsedTime = elapsedTime+totaltime;
    cnt++;
  }
  // for(int g=0;g<ngpu;g++){
  //   cudaEventDestroy(e_start[g]);
  //   cudaEventDestroy(e_stop[g]); 
  // }
  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);
  return elapsedTime/8;
  
}

template<class T>
long long CheckError(T* x, long long data_size){
  long long cnt = 0;
  // int a;
  for(long long i=0;i<data_size;i++){
    if(x[i]!=(T)(i*ngpu)){
      cnt++;
      // printf("%lld %lf\n", i, x[i]);
      // scanf("%d", &a);
    }
  }
  return cnt;
}

template<class T>
void GenData(T* &x, long long data_size){
  x   = (T*)malloc(sizeof(T)*data_size);
  for(long long i = 0; i < data_size; i++){ 
    x[i] = (T)(i);
  }
}

int main(int argc, char* argv[]) {

  int loop = -1, split =-1;
  if(argc == 1){
    printf("Argument Error\n");
    return 1;
  }
  if(argc > 1)loop = atoi(argv[1]);
  if(argc > 2)split = 1;
  double *x;
  // 4,999,900,001 2,147,483,648
  long long data_size = 2147483648;//16GB
  GenData(x,data_size);
  ncclComm_t* comm;
  cudaStream_t* s;
  int* device;
  NCCLInit(comm, s, device);
  printf("ncclInit\n");
  double**device_src = nullptr, **device_dst = nullptr;
  CudaMemInit(device_src, data_size, device);
  CudaMemInit(device_dst, data_size, device);
  printf("CudaMemInit\n");
  for(int g=0;g<ngpu;g++){
    cudaSetDevice(device[g]);
    cudaMemcpy(device_src[g],  x, data_size * sizeof(double), cudaMemcpyHostToDevice); /*Copy from Host to Devices*/
  }

  float time;
  size_t mode = 0;
  clock_t begin, end;

  // printf("******   warm up mode = %d ******\n", mode);
  // allreduceTest(data_size, device, 5, comm, s, device_src, device_dst, mode);
  // printf("******************************\n");

  // begin = clock();
  // time = allreduceTest(data_size, device, loop, comm, s, device_src, device_dst, mode);
  // end = clock();
  // cudaSetDevice(device[0]);
  // cudaMemcpy(x,  device_dst[0], data_size * sizeof(double), cudaMemcpyDeviceToHost);
  // printf("Total ERROR %lld\n", CheckError(x, data_size));
  // printf("kernel cost %3.1f ms\n", time);
  // printf("Total time %lf s\n", (double)(end - begin) / CLOCKS_PER_SEC);
  // printf("******************************\n\n");


  // mode = 1;
  // printf("******   warm up mode = %d ******\n", mode);
  // allreduceTest(data_size, device, 5, comm, s, device_src, device_dst, mode);
  // printf("******************************\n");

  // begin = clock();
  // time = allreduceTest(data_size, device, loop, comm, s, device_src, device_dst, mode);
  // end = clock();
  // cudaSetDevice(device[0]);
  // cudaMemcpy(x,  device_dst[0], data_size * sizeof(double), cudaMemcpyDeviceToHost);
  // printf("Total ERROR %lld\n", CheckError(x, data_size));
  // printf("kernel cost %3.1f ms\n", time);
  // printf("Total time %lf s\n", (double)(end - begin) / CLOCKS_PER_SEC);
  // printf("******************************\n\n");

  mode = 7;
  printf("******   warm up mode = %zu ******\n", mode);
  allreduceTest(data_size, device, 5, comm, s, device_src, device_dst, mode);
  printf("******************************\n");

  begin = clock();
  time = allreduceTest(data_size, device, loop, comm, s, device_src, device_dst, mode);
  end = clock();
  cudaSetDevice(device[0]);
  cudaMemcpy(x,  device_dst[0], data_size * sizeof(double), cudaMemcpyDeviceToHost);
  printf("Total ERROR %lld\n", CheckError(x, data_size));
  printf("kernel cost %3.1f ms\n", time);
  printf("Total time %lf s\n", (double)(end - begin) / CLOCKS_PER_SEC);
  printf("******************************\n\n");



  ncclfinish(comm, s, device);
  cudaFree(device_src);
  cudaFree(device_dst);
  return 0;

}/*main*/




