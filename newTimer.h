#pragma once
#include "stdio.h"
#include <cuda_runtime.h>
#include <sys/times.h> //for struct tms
#include "unistd.h" // for sysconf function and _SC_CLK_TCK
namespace newTimer{
	struct TimerGPU;
	struct TimerCPU;
	extern struct TimerGPU * GPUTimers;
	extern struct TimerCPU * CPUTimers;
	extern cudaEvent_t start, stop;
	extern long CPUTICKCLOCK;
	void timeCPUZero(const char *, const char *, const char *, const int, const char *);
	void timeCPUOne(const char *, const int, const char *);
	void timeGPUZero(const char *, const char *, const char *, const int, const char *);
	void timeGPUOne(const char *, const int, const char *);

	struct TimerGPU * timerGPUSearch(const char *, struct TimerGPU *); //search timers with name wether or not exist
	struct TimerCPU * timerCPUSearch(const char *, struct TimerCPU *); //search timers with name wether or not exist
	struct TimerGPU * getRearTimerGPU(struct TimerGPU *); //Get the rear timer
	struct TimerCPU * getRearTimerCPU(struct TimerCPU *); //Get the rear timer
	void outputTimers(); //output all of performance of kernels
	char *  itoa(int);
	void setTimerName(char * timerName, char * kernelName);
	void setTimerGPUName(struct TimerGPU *, const char *);
	void setTimerCPUName(struct TimerCPU *, const char *);
	bool timersCompare(const char * timerName, const char * kernelName);
	//Macro simplifies function
	#define TIMERGPU0(str1, str2) (timeGPUZero(str1, str2, __func__, __LINE__, __FILE__))
	#define TIMERGPU1(str1) (timeGPUOne(str1, __LINE__, __FILE__))
	#define TIMERCPU0(str1, str2) (timeCPUZero(str1, str2, __func__, __LINE__, __FILE__))
	#define TIMERCPU1(str1) (timeCPUOne(str1, __LINE__, __FILE__))
}
