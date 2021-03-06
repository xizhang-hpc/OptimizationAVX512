#include "stdio.h"
#include "runLoadFlux.h"
#include "GlobalVariablesHost.h"
//#include "GPULoadFlux.h"
#include "HostLoadFlux.h"
#include "AVX512LoadFlux.h"
#include "constantVars.hxx"
//#include "Validation.h"
void runLoadFlux(){
	preProcessLoadFlux();
	int localStart, localEnd;
	int loopID;
	int LOOPNUM = 500; //define the local LOOPNUM
	for (loopID = 0; loopID < LOOPNUM; loopID++){
		localStart = 0;
	    	do
 		{   
        		//localEnd = localStart + SEG_LEN;
        		localEnd = localStart + nTotalFace;
		        if (localEnd > nTotalFace) {   
		        	localEnd = nTotalFace;
        		}       
			HostFaceLoopLoadFlux(localStart, localEnd, loopID);
			AVX512FaceLoopLoadFluxOutside(loopID);
			//AVX512FaceLoopLoadFluxInside(loopID);
			//AVX512FaceLoopLoadFluxInsideReOpt(loopID);
			AVX512FaceLoopLoadFluxOutsideReOpt(loopID);
			HostCellLoopLoadFlux(loopID);
			AVX512CellLoopLoadFluxOutside(loopID);
			//AVX512CellLoopLoadFluxInside(loopID);
			localStart = localEnd;
	    	} while (localStart < nTotalFace);
	}
}

void preProcessLoadFlux(){
	printf("preProcessLoadFlux\n");	
	//for host data
	setPropertyOfSimulation();
	readLeftRightCellofFace();
	readNumberFaceOfEaceCell();
	readCell2Face();
	setLeftRightFace();
	setSEGLEN();
	readBoundaryType();
	mallocRes();
	setResRandom();
	setResOrg();
	mallocFlux();
	setFluxRandom();
	faceColor();
	reorderFaceVars();
}
