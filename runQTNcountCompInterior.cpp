#include "stdio.h"
#include "stdlib.h"
#include "runQTNcountCompInterior.h"
#include "HostInteriorFaceNCountQNodeTNodeCal.h"
#include "AVX512Interpolation.h"
#include "GlobalVariablesHost.h"
#include "constantVars.hxx"
//#include "Validation.h"
void preProcessQTNcountCompInterior(){
	printf("preProcessQTNcountCompInterior\n");
	//for host variables
	readLeftRightCellofFace();
	readBoundaryType();
	readNumberNodeOfEachFace();
	readFace2Node();
	readNumberNodeOfEachCell();
	readCell2Node();
	readNumberCellOfEachNode();
	readNode2Cell();
	setPropertyOfSimulation();
	setQNSRandom();
	setTCellRandom();
	mallocQNode();
	mallocTNode();
	mallocNCount();
	setFaceNumberOfEachNode();
	setNode2Face();
/*
	//for device variables
	devAltCpyLeftRightCellofFace();
	devAltCpyBoundaryType();
	devAltCpyNodeNumberOfEachFace();
	devAltCpyFace2Node();
	devAltCpyCellNumberOfEachNode();
	devAltCpyNode2Cell();
	devAltCpyNodeNumberOfEachCell();
	devAltCpyCell2Node();
	devAltCpyQNS();
	devAltCpyTCell();
	devAltQNode();
	devAltTNode();
	devAltNCount();
*/
	//for cancel the branch
	//setIsGhostCell();
	//for get real nCount
	setNode2CellCount();
	//devAltCpyNode2CellCount();
	setCell2NodeCount();
	faceColorByNode();
	reorderFaceVarsByNode();
	cellColorByNode();
	reorderCellVarsByNode();
	//devAltCpyCell2NodeCount();
	//for test
	printf("nTotalFace = %d\n", nTotalFace);
	printf("nTotalCell = %d\n", nTotalCell);
	printf("nBoundFace = %d\n", nBoundFace);
	printf("nTotalNode = %d\n", nTotalNode);
	printf("nl = %d, nchem = %d, nTemperature = %d\n", nl, nchem, nTemperature);
}

void runQTNcountCompInterior(){
	//preProcessQTNcountCompInterior();
	//int loopID;
	//loopID = 0;
	//initNCountQNodeTNodeZero(loopID);
	//CallHostInteriorFaceNCountQNodeTNodeCal(loopID);
	//CallHostFaceNCountQNodeTNodeCal(loopID);
	//CallHostNodeLoopNCountQNodeTNodeCal(loopID);
	//CallHostCellLoopNCountQNodeTNodeCal(loopID);
	//HostAverageQNodeTNode(loopID);
	//setNcountQNodeTNodeZeroDevice(loopID);
	//CallGPUInteriorFaceNCountQNodeTNodeCal(loopID);
	//CallGPUFaceNCountQNodeTNodeCal(loopID);
	//CallGPUNodeLoopNCountQNodeTNodeCal(loopID);
	//CallGPUCellLoopNCountQNodeTNodeCal(loopID);
	//validateQTNcountCompInterior();
	//validateQNodeTNode(0, "CellLoop");

		
	preProcessQTNcountCompInterior();
	int loopID;
	int loopNum = 1000;
	for (loopID = 0; loopID < loopNum; loopID++){
	//for (loopID = 0; loopID < 10; loopID++){
		if ( loopID % (loopNum/10) == 0) printf("%.4f%%\n", (float)(loopID*100.0/loopNum));
		//CallHostNodeLoopNCountQNodeTNodeCalFinal(loopID);
		CallHostCellLoopNCountQNodeTNodeCalFinal(loopID);
		AVX512CellLoopSepInterpolation(loopID);
		CallHostFaceNCountQNodeTNodeCal(loopID);
		AVX512FaceLoopSepInterpolation(loopID);
	}
	
}
