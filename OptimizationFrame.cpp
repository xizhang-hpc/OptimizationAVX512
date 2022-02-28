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
	runQTNcountCompInterior(); //Optimize GPUInteriorFaceNCountQNodeTNodeCal
	//runCompGradientGGNodeFaceCal();//Optimize CompGradientGGNodeInteriorFaceCal
//	runLoadFlux(); //Optimize LoadFlux
	//runLocalMinMax(); //local max and min of qNS
	outputTimers();
	//free host and device variables
	freeGlobalVariablesHost();
	//freeGlobalVariablesDevice();
	printf("Optimize finish\n");
	return 0;
}
