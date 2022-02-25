#include "stdio.h"
#include "stdlib.h"
#include "HostLocalMinMax.h"
#include "GlobalVariablesHost.h"
#include "newTimer.h"
#include "immintrin.h"
#define N_UNROLL 8
using namespace newTimer;
void HostCellLoopLocalMinMax(const int loopID){
	if (loopID == 0) printf("HostCellLoopLocalMinMax\n");
	//set dMin and dMax by qNS
	setDMaxDMinByQNS(loopID);
	int cellID, faceID;
	int numFaces;
	int le, re, offset;

	//cell loop with the aid of cell2Cells
	setDMaxDMinByQNS(loopID);
	int cellIDOffset;
	TIMERCPU0("HostCellLoopMinMax", "Cell Loop MinMax");
	for (cellID = 0; cellID < nTotalCell; cellID++){
		numFaces = faceNumberOfEachCell[cellID];
		for (offset = 0; offset < numFaces; offset++){
			cellIDOffset = cell2Cell[cellID][offset];
                	dMin[cellID] = MIN(dMin[cellID], qNS[0][cellIDOffset]);
	                dMax[cellID] = MAX(dMax[cellID], qNS[0][cellIDOffset]);
		}
	}
	TIMERCPU1("HostCellLoopMinMax");

}

void HostFaceLoopLocalMinMax(const int loopID){
	if (loopID == 0) printf("HostFaceLoopLocalMinMax\n");
	int iFace;
	int le, re;
	//set dMin and dMax by qNS
	setDMaxDMinByQNS(loopID);
	//for boundary faces
	TIMERCPU0("HostFaceLoopMinMax", "Face Loop MinMax");
        for (iFace = 0; iFace < nBoundFace; ++ iFace){
                le       = leftCellofFace[iFace];
                re       = iFace + nTotalCell;

                dMin[le] = MIN(dMin[le], qNS[0][re]);
                dMax[le] = MAX(dMax[le], qNS[0][re]);
        }
	
	//for interior faces
        for (iFace = nBoundFace; iFace < nTotalFace; ++ iFace)
        {
                int le,re;
                le       = leftCellofFace [iFace];
                re       = rightCellofFace[iFace];
                dMin[le] = MIN(dMin[le], qNS[0][re]);
                dMax[le] = MAX(dMax[le], qNS[0][re]);

                dMin[re] = MIN(dMin[re], qNS[0][le]);
                dMax[re] = MAX(dMax[re], qNS[0][le]);
        }
	TIMERCPU1("HostFaceLoopMinMax");

}

void setDMaxDMinByQNS(const int loopID){
	if (loopID == 0) printf("setDMaxDMinByQNS\n");
	int iCell;
	for (iCell = 0; iCell < nTotalCell; iCell++) {
		dMin[iCell] = qNS[0][iCell];
		dMax[iCell] = qNS[0][iCell];
	}
}
fpkind MIN(fpkind a, fpkind b){
        return(a>b?b:a);
}
fpkind MAX(fpkind a, fpkind b){
        return(a>b?a:b);
}
