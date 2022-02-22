#include "newTimer.h"
#include "string.h"
#include "stdlib.h"
namespace newTimer{
	struct TimerCPU{
		const char * timerName; //The name of timer
		const char * timerDescription;
		const char * parentFuncName; //store the name of function, important
		int codeLineBegin;
		const char * codePositionBegin;
		int codeLineEnd;
		const char * codePositionEnd;
		int  count;
		bool timerOpen;
		double timerStart; //for storing the beginning time stamp, gpu timer does not own.
		double totalTime;
		struct TimerCPU * pNext; 
	};

	struct TimerCPU * CPUTimers;
	cudaEvent_t start, stop;
	long CPUTICKCLOCK = 0;

	void timeCPUZero(const char * kernelName, const char * kernelDescription, const char * funcName, const int codeLine, const char * codePosi){
		#ifdef TIMERDEBUG
			printf("begin of timeCPUZero\n");
		#endif
		if (CPUTimers) {
		//GPUTimers exist!
			//search link
			struct TimerCPU * pTimer = timerCPUSearch(kernelName, CPUTimers); 
			//the kiernelName already exists in GPUTimers.
			if (pTimer) {
				#ifdef TIMERDEBUG
					printf("In if (pTimer) pTimer->timerName = %s, kernelName=%s\n", pTimer->timerName, kernelName);
				#endif
				//add end
				//timerOpen has already been opened.
				if (pTimer->timerOpen) {
					printf("Error: timerOpen in %s, should not be open, which is in line %d of %s.\n", pTimer->timerName, pTimer->codeLineBegin, pTimer->codePositionBegin);
					exit(1);
				}else{
					(pTimer->count)++;
					pTimer->timerOpen = true;
					#ifdef CLOCKMODE
						//record cpu time by clock()
						pTimer->timerStart = clock();
					#else
						struct tms timerBegin;
						times(&timerBegin);
				 	    pTimer->timerStart = timerBegin.tms_utime 
						+ timerBegin.tms_stime;
					#endif
					#ifdef TIMERDEBUG
						printf("In If: pTimer->timerStart = %f\n", pTimer->timerStart);
				#endif
				}
			}
			//the new timer kiernelName should be added.
			else{
				//struct Timer newGPUTimer;
				struct TimerCPU * newCPUTimer = (struct TimerCPU *)malloc(sizeof(struct TimerCPU));
				//noting that when timerName is changed into char *, you cannot just use the pointer, because it is not a continuous space, which is just a pointer. Thus, the pointer of newGPUTimer should be passed into the function, or the address of newGPUTimer->timerName.
				setTimerCPUName(newCPUTimer, kernelName);
				newCPUTimer->timerDescription = kernelDescription;
				newCPUTimer->parentFuncName = funcName;
				newCPUTimer->codeLineBegin = codeLine;
				newCPUTimer->codePositionBegin = codePosi;
				newCPUTimer->count = 1;
				newCPUTimer->timerOpen = true;
				newCPUTimer->totalTime = 0.0;
				newCPUTimer->pNext = NULL;
				pTimer = getRearTimerCPU(CPUTimers); 
				#ifdef TIMERDEBUG
					printf("In else  pTimer->timerName = %s, kernelName=%s\n", pTimer->timerName, kernelName);
				#endif
				//pTimer->pNext = &newGPUTimer;
				pTimer->pNext = newCPUTimer;
				#ifdef CLOCKMODE
					//record cpu time by clock()
					newCPUTimer->timerStart = clock();
				#else
					struct tms timerBegin;
					times(&timerBegin);
					newCPUTimer->timerStart = timerBegin.tms_utime 
						+ timerBegin.tms_stime;
				#endif
				#ifdef TIMERDEBUG
					printf("In If else: newCPUTimer->timerStart = %f\n", newCPUTimer->timerStart);
				#endif
			}
		} 
		//GPUTimers do not exist! The first time it is used.
		else {
			#ifdef TIMERDEBUG
				printf("Create timers the first node\n");
			#endif
		       	
			//define CLOCKMODE
			#ifdef CLOCKMODE
				CPUTICKCLOCK = CLOCKS_PER_SEC;
			#else
				CPUTICKCLOCK = sysconf(_SC_CLK_TCK);
			#endif
			if (!CPUTICKCLOCK) {
				printf("Error: CPUTICKCLOCK = 0\n");
				exit(1);
			}
			//struct Timer newGPUTimer;
			struct TimerCPU * newCPUTimer = (struct TimerCPU *)malloc(sizeof(struct TimerCPU));
			//noting that when timerName is changed into char *, you cannot just use the pointer, because it is not a continuous space, which is just a pointer. Thus, the pointer of newGPUTimer should be passed into the function, or the address of newGPUTimer->timerName.
			setTimerCPUName(newCPUTimer, kernelName);
			#ifdef TIMERDEBUG
				printf("newCPUTimer->timerName = %s\n", newCPUTimer->timerName);
			#endif
			newCPUTimer->timerDescription = kernelDescription;
			newCPUTimer->parentFuncName = funcName;
			newCPUTimer->codeLineBegin = codeLine;
			newCPUTimer->codePositionBegin = codePosi;
			newCPUTimer->count = 1;
			newCPUTimer->timerOpen = true;
			newCPUTimer->totalTime = 0.0;
			newCPUTimer->pNext = NULL;
			CPUTimers = newCPUTimer;
			#ifdef CLOCKMODE
				//record cpu time by clock()
				newCPUTimer->timerStart = clock();
			#else
				struct tms timerBegin;
				times(&timerBegin);
				newCPUTimer->timerStart = timerBegin.tms_utime 
					+ timerBegin.tms_stime;
			#endif
			#ifdef TIMERDEBUG
				printf("In else: newCPUTimer->timerStart = %f\n", newCPUTimer->timerStart);
			#endif
		}
		#ifdef TIMERDEBUG
			printf("end of timeCPUZero\n");
		#endif
	}



	void timeCPUOne(const char * kernelName, const int codeLine, const char * codePosi){
		#ifdef TIMERDEBUG
			printf("begin of timeCPUOne\n");
		#endif
		#ifdef CLOCKMODE
			//record cpu time by clock()
			double timerEnd = clock();
		#else
			struct tms timerStop;
			times(&timerStop);
			double timerEnd = timerStop.tms_utime + 
				timerStop.tms_stime;
		#endif
		#ifdef TIMERDEBUG
			printf("In timeCPUOne, timerEnd = %f\n", timerEnd);
		#endif
		//CPUTimers do not exist
		if (!CPUTimers) {
			printf("Error: CPUTimers do not exsit!\n");
			exit(1);
		}
				
		struct TimerCPU * pTimer = timerCPUSearch(kernelName, CPUTimers);	
		if (pTimer) {
			//calculate ELAPSEDTIME by timerStart and timerEnd
			double ELAPSEDTIME = (timerEnd - pTimer->timerStart)/CPUTICKCLOCK;
			#ifdef TIMERDEBUG
				printf("kernelName = %s, pTimer->timerName= %s\n", kernelName, pTimer->timerName);
			#endif
			pTimer->timerOpen = false;
			pTimer->totalTime += ELAPSEDTIME;
			#ifdef TIMERDEBUG
				printf("In timeCPUOne, kernelName =%s, ELAPSEDTIME = %13.7f, pTimer->totalTime = %13.7f\n", kernelName, ELAPSEDTIME, pTimer->totalTime);
			#endif
			pTimer->codeLineEnd = codeLine;
			pTimer->codePositionEnd = codePosi;
		}
		else {
			printf("Error: %s, is not in CPUTimers, which is in %d of %s.\n", kernelName, codeLine, codePosi);
			exit(1);
		}	
		#ifdef TIMERDEBUG
			printf("end of timeCPUOne\n");
		#endif
	}
	

	struct TimerCPU *  timerCPUSearch(const char * kernelName, struct TimerCPU * pTimer){
		while (pTimer){
			#ifdef TIMERDEBUG
				printf("In timerGPUSearch, pTimer->timerName = %s, kernelName = %s\n", pTimer->timerName, kernelName);
			#endif
			if (timersCompare(pTimer->timerName, kernelName)) {
				#ifdef TIMERDEBUG
					printf("In timerGPUSearch, for timerComprre, pTimer->timerName = %s is equal to kernelName = %s\n", pTimer->timerName, kernelName);
				#endif
				return pTimer; 
			}
			else pTimer = pTimer->pNext;
		}
		return NULL;
	}


	struct TimerCPU *  getRearTimerCPU(struct TimerCPU * pTimer){
		while (pTimer->pNext){
			pTimer = pTimer->pNext;
		}
		return pTimer;
	}

	void outputTimers(){
	      #ifdef GPUTIME
		if (GPUTimers){
			//The performance is also output into file performanceGPU
			FILE * fpGPU = fopen("performanceGPU","w");
			struct TimerGPU * pTimer = GPUTimers; 
			int nTimers = 0;
			printf("*************Performance of GPU Kernels*************\n");
			fprintf(fpGPU, "*************Performance of GPU Kernels*************\n");
			printf("No.	Name		Parent	   Elapsed Time	   frequency\n");
			fprintf(fpGPU, "No.	Name		Parent	   Elapsed Time	   frequency\n");
			while(pTimer){
				nTimers++;
				printf("%d	%s	%s	%f	%d\n", nTimers, pTimer->timerName, pTimer->parentFuncName, pTimer->totalTime, pTimer->count);
				fprintf(fpGPU, "%d	%s	%s	%f	%d\n", nTimers, pTimer->timerName, pTimer->parentFuncName, pTimer->totalTime, pTimer->count);
				pTimer = pTimer->pNext;
			}
			pTimer = GPUTimers;
			printf("*************Description of GPU Kernels*************\n");
			fprintf(fpGPU, "*************Description of GPU Kernels*************\n");
			printf("No.	Name		Description	File:Line(Begin)	File:Line(End)\n");
			fprintf(fpGPU, "No.	Name		Description	File:Line(Begin)	File:Line(End)\n");
			nTimers = 0;
			while(pTimer){
				nTimers++;
				printf("%d	%s	%s	%s:Line%d	%s:Line%d\n", nTimers, pTimer->timerName, pTimer->timerDescription, pTimer->codePositionBegin, pTimer->codeLineBegin, pTimer->codePositionEnd, pTimer->codeLineEnd);
				fprintf(fpGPU, "%d	%s	%s	%s:Line%d	%s:Line%d\n", nTimers, pTimer->timerName, pTimer->timerDescription, pTimer->codePositionBegin, pTimer->codeLineBegin, pTimer->codePositionEnd, pTimer->codeLineEnd);
				pTimer = pTimer->pNext;
			}
			pTimer = GPUTimers;
			while(pTimer){
				struct TimerGPU * pDestroy = pTimer;
				pTimer = pTimer->pNext;
				free(pDestroy);
			}	
			printf("%d gpu kernels are recorded\n", nTimers);
			fprintf(fpGPU, "%d gpu kernels are recorded\n", nTimers);
			fclose(fpGPU);
		}
	     #endif
		if (CPUTimers){
			//The performance is also output into file performanceCPU
			FILE * fpCPU = fopen("performanceCPU","w");
			struct TimerCPU * pTimer = CPUTimers; 
			int nTimers = 0;
			printf("*************Performance of CPU Kernels*************\n");
			fprintf(fpCPU, "*************Performance of CPU Kernels*************\n");
			printf("No.	Name		Parent	   Elapsed Time	   frequency\n");
			fprintf(fpCPU, "No.	Name		Parent	   Elapsed Time	   frequency\n");
			while(pTimer){
				nTimers++;
				printf("%d	%s	%s	%f	%d\n", nTimers, pTimer->timerName, pTimer->parentFuncName, pTimer->totalTime, pTimer->count);
				fprintf(fpCPU, "%d	%s	%s	%f	%d\n", nTimers, pTimer->timerName, pTimer->parentFuncName, pTimer->totalTime, pTimer->count);
				pTimer = pTimer->pNext;
			}
			pTimer = CPUTimers;
			printf("*************Description of CPU Kernels*************\n");
			fprintf(fpCPU, "*************Description of CPU Kernels*************\n");
			printf("No.	Name		Description	File:Line(Begin)	File:Line(End)\n");
			fprintf(fpCPU, "No.	Name		Description	File:Line(Begin)	File:Line(End)\n");
			nTimers = 0;
			while(pTimer){
				nTimers++;
				printf("%d	%s	%s	%s:Line%d	%s:Line%d\n", nTimers, pTimer->timerName, pTimer->timerDescription, pTimer->codePositionBegin, pTimer->codeLineBegin, pTimer->codePositionEnd, pTimer->codeLineEnd);
				fprintf(fpCPU, "%d	%s	%s	%s:Line%d	%s:Line%d\n", nTimers, pTimer->timerName, pTimer->timerDescription, pTimer->codePositionBegin, pTimer->codeLineBegin, pTimer->codePositionEnd, pTimer->codeLineEnd);
				pTimer = pTimer->pNext;
			}
			pTimer = CPUTimers;
			while(pTimer){
				struct TimerCPU * pDestroy = pTimer;
				pTimer = pTimer->pNext;
				free(pDestroy);
			}	
			printf("%d cpu kernels are recorded\n", nTimers);
			fprintf(fpCPU, "%d cpu kernels are recorded\n", nTimers);
			fclose(fpCPU);
		}
	}

	char * itoa(int number){
		if (number < 0) {
			printf("Error: %d is a negative number\n", number);
		}
		static char result[100];
		int posi = 0;
		int i;
		while((int(number%10)>0)||(number/10>0)||(posi==0)){
			result[posi] = (number % 10) + '0' ;
			number = number/10;
			posi++;
		}
		result[posi] = '\0';
		for (i = 0; i < int(posi/2); i++){
			char temp = result[i];
			result[i] = result[posi-1-i];
			result[posi-1-i] = temp;
		}
		return result;
	}

	void setTimerName(char * timerName, char * kernelName){ //need to improve
		int posi = 0;
		int i = 0;
		while(kernelName[posi] != '\0'){
			posi++;
		}
		//kernelName only has 0 element
		if (posi == 0) {
			printf("Error: kernelName is empty\n");
			exit(1);
		}
		char * kernelNameCopy = (char *)malloc(posi * sizeof(char)); 
		for(i = 0; i < posi; i++) kernelNameCopy[i] = kernelName[i];
		kernelNameCopy[posi] = '\0';
		timerName = kernelNameCopy;
	}


	void setTimerCPUName(struct TimerCPU * newTimer, const char * kernelName){ //need to improve
		int posi = 0;
		int i = 0;
		while(kernelName[posi] != '\0'){
			posi++;
		}
		//kernelName only has 0 element
		if (posi == 0) {
			printf("Error: kernelName is empty\n");
			exit(1);
		}
		char * kernelNameCopy = (char *)malloc(posi * sizeof(char)); 
		for(i = 0; i < posi; i++) kernelNameCopy[i] = kernelName[i];
		kernelNameCopy[posi] = '\0';
		newTimer->timerName = kernelNameCopy;
		#ifdef TIMERDEBUG
			printf("kernelNameCopy = %s, timerName = %s\n", kernelNameCopy, newTimer->timerName);
		#endif
	}

	//bool timersCompare(char * timerName, char * kernelName){
	bool timersCompare(const char * timerName, const char * kernelName){
		bool value = true;
		int posiTimerName = 0;
		int posiKernelName = 0;
		int i;
		while(timerName[posiTimerName] != '\0'){
			posiTimerName++;
		}
		while(kernelName[posiKernelName] != '\0'){
			posiKernelName++;
		}
		if (posiTimerName != posiKernelName) {
			value = false;
			return value;
		}
		for (i=0; i < posiKernelName; i++) if (timerName[i] != kernelName[i]) value = false;
		return value;
	}


}
