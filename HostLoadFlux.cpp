#include "stdio.h"
#include "HostLoadFlux.h"
#include "GlobalVariablesHost.h"
//#include "newTimer.h"
//using namespace newTimer;
void resetRes(const int loopID){
	if (loopID == 0) printf("resetRes\n");
	int equationID, cellID;
	int nEquation = nl + nchem;
	int nTotal = nTotalCell + nBoundFace;
	for (cellID = 0; cellID < nTotal; cellID++){
		for (equationID = 0; equationID < nEquation; equationID++){
			res[equationID][cellID] = resOrg[equationID][cellID];
		}
	}
}
void HostFaceLoopLoadFlux(const int localStart, const int localEnd, const int loopID){
	if (loopID == 0) printf("HostFaceLoopLoadFlux\n");
	int nMid;
	if (localEnd <= nBoundFace) nMid = localEnd; //all of faces on boundary
	else {
		if (localStart > nBoundFace) nMid = localStart; //all of faces on interior faces.
		else nMid = nBoundFace; //one part on boundary, and one part on interior faces.
	}
	int faceID, equationID;
	int le, re;
	int nEquation = nl + nchem;
	//reser Res by ResOrg
	resetRes(loopID);
//	TIMERCPU0("HostFaceLoopLoadFlux", "Face Loop with no Optimization");
	//Faces on boundary faces
	for (faceID = localStart; faceID < nMid; faceID++){
		le = leftCellofFace[faceID];
		for (equationID = 0; equationID < nEquation; equationID++){
			res[equationID][le] -= flux[equationID][faceID];
		}
	}
	//Faces on interior faces
	for (faceID = nMid; faceID < localEnd; faceID++){
		le = leftCellofFace[faceID];
		re = rightCellofFace[faceID];
		for (equationID = 0; equationID < nEquation; equationID++){
			res[equationID][le] -= flux[equationID][faceID];
			res[equationID][re] += flux[equationID][faceID];
		}
	}
//	TIMERCPU1("HostFaceLoopLoadFlux");
}
void HostCellLoopLoadFlux(const int loopID){
	if (loopID == 0) printf("HostCellLoopLoadFlux\n");
	int cellID, equationID;
	int faceID, numFacesInCell, offset, leftRight;
	int nEquation = nchem + nl;
	//reser Res by ResOrg
	resetRes(loopID);
//	TIMERCPU0("HostCellLoopLoadFluxUseIf", "Cell Loop with branch");
	for (cellID = 0; cellID < nTotalCell; cellID++){
		numFacesInCell = faceNumberOfEachCell[cellID];
		for (offset = 0; offset < numFacesInCell; offset++){
			faceID = cell2Face[cellID][offset];
			leftRight = leftCellofFace[faceID];
			for (equationID = 0; equationID < nEquation; equationID++){
				if (leftRight == cellID) res[equationID][cellID] -= flux[equationID][faceID];
				else  res[equationID][cellID] += flux[equationID][faceID]; 
			}
		}
	}
//	TIMERCPU1("HostCellLoopLoadFluxUseIf");
}
