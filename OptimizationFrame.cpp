#include "stdio.h"
//#include "OptimizationFrame.h"
//#include "DeviceControl.h"
#include "GlobalVariablesHost.h"
#include "runQTNcountCompInterior.h"
//#include "runCompGradientGGNodeFaceCal.h"
#include "runLoadFlux.h"
#include "runLocalMinMax.h"
#include "newTimer.h"
using namespace newTimer;
int main(int argc, char ** argv){
	printf("Hello, Optimization Platform\n");
//	deviceControl();
    #ifdef DATAINTERPOLATE
        	runQTNcountCompInterior(); //Optimize GPUInteriorFaceNCountQNodeTNodeCal
    #endif
	#ifdef FLUXSUM
		runLoadFlux(); //Optimize LoadFlux
	#endif
	#ifdef MAXMIN
		runLocalMinMax(); //local max and min of qNS
	#endif
	outputTimers();
	//free host and device variables
	freeGlobalVariablesHost();
	//freeGlobalVariablesDevice();
	printf("Optimize finish\n");
	return 0;
}
