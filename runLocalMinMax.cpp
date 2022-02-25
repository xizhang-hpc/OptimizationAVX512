#include "constantVars.hxx"
#include "runLocalMinMax.h"
#include "HostLocalMinMax.h"
#include "GlobalVariablesHost.h"
void runLocalMinMax(){
	preProcessLocalMinMax();
	int loopID;
	int loopNum = 100;
	for (loopID = 0; loopID < loopNum; loopID++){
		HostFaceLoopLocalMinMax(loopID);
		HostCellLoopLocalMinMax(loopID);
	}
}

void preProcessLocalMinMax(){
	//set basic calculation property
	setPropertyOfSimulation();
	//host variables
	readLeftRightCellofFace();
	readNumberFaceOfEaceCell();
	readCell2Face();
	setCell2Cell();
	readBoundaryType();
	setQNSRandom();
	mallocDMinDMax();
	
}
