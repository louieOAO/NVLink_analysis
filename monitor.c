#include<stdio.h>
#include<unistd.h>
#include<string.h>
#include <stdlib.h>
#include"/usr/local/cuda-12.1/include/nvml.h"

#define NVLINK_NUM 12


typedef struct counterinfo{
    unsigned long long txtimestamp;
    unsigned long long rxtimestamp;
    unsigned long long rxcounter;
    unsigned long long txcounter;

}CounterInfo;

nvmlReturn_t GetNVLinkCounter(int device_id, nvmlDevice_t device, CounterInfo* now){
    nvmlReturn_t result;
    nvmlFieldValue_t values[NVLINK_NUM*2];

    for(int i=0;i<NVLINK_NUM;i++){
        values[i*2].scopeId = i;
        values[i*2].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;

        values[i*2+1].scopeId = i;
        values[i*2+1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
    }

    nvmlDeviceGetFieldValues(device, NVLINK_NUM*2, values);

    for(int i=0;i<NVLINK_NUM;i++){
        now[i].rxcounter   = values[i*2].value.ullVal;
        now[i].rxtimestamp = values[i*2].timestamp;

        now[i].txcounter   = values[i*2+1].value.ullVal;
        now[i].txtimestamp = values[i*2+1].timestamp;
    }


    return result;
}


void ComputeCounterDiff(CounterInfo* prev, CounterInfo* now, int gpu_id, int printchange){
    
    for(int i=0;i<NVLINK_NUM;i++){
        if(now[i].rxcounter != prev[i].rxcounter){
            long long diff = (now[i].rxcounter - prev[i].rxcounter);
            long long timediff = now[i].rxtimestamp - prev[i].rxtimestamp;
            if(printchange)printf("%lld id: %d link: %d Rx: %lld\n", now[i].rxtimestamp, gpu_id, i, diff);
        }

        if(now[i].txcounter != prev[i].txcounter){
            long long diff = (now[i].txcounter - prev[i].txcounter);
            long long timediff = now[i].txtimestamp - prev[i].txtimestamp;
            if(printchange)printf("%lld id: %d link: %d Tx: %lld\n", now[i].txtimestamp, gpu_id, i, diff);  
        }
        prev[i].txcounter = now[i].txcounter;
        prev[i].rxcounter = now[i].rxcounter;
        prev[i].txtimestamp = now[i].txtimestamp;
        prev[i].rxtimestamp = now[i].rxtimestamp;
    }
    

}



int main(int argc, char** argv){
    int device_id[10], ngpu = 0;
    if(argc<2){
        printf("Please enter GPU id\n");
    }
    if(argc == 2){
        const char split[1] = {','};
        char* token = strtok(argv[1], split);
        for(ngpu=0;token!=NULL;ngpu++){
            device_id[ngpu] = atoi(token);
            token = strtok(NULL, split);
        }
    }

    nvmlDevice_t device[ngpu];
    CounterInfo now[ngpu][NVLINK_NUM], prev[ngpu][NVLINK_NUM];
    nvmlInit();
    for(int i=0;i<ngpu;i++){
        nvmlDeviceGetHandleByIndex(device_id[i], &device[i]);
    }
    sleep(1);
    for(int i=0;i<ngpu;i++){
        GetNVLinkCounter(device_id[i], device[i], now[i]);
        ComputeCounterDiff(prev[i], now[i], device_id[i], 0);
    }
    while(1){
        for(int i=0;i<ngpu;i++){
            GetNVLinkCounter(device_id[i], device[i], now[i]);
            ComputeCounterDiff(prev[i], now[i], device_id[i], 1);
        }
        fflush(stdout);
    }


}