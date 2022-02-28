#include "constantVars.hxx"
#include "runLocalMinMax.h"
#include "HostLocalMinMax.h"
#include "GlobalVariablesHost.h"
#include "AVX512LocalMinMax.h"
void runLocalMinMax(){
	preProcessLocalMinMax();
	int loopID;
	int loopNum = 1000;
	for (loopID = 0; loopID < loopNum; loopID++){
		HostFaceLoopLocalMinMax(loopID);
		HostCellLoopLocalMinMax(loopID);
		AVX512FaceLoopLocalMinMax(loopID);
		AVX512FaceLoopLocalMinMaxReOpt(loopID);
		AVX512CellLoopLocalMinMax(loopID);
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
	faceColor();
	reorderFaceVars();
	
}
