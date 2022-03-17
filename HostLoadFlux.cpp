#include "stdio.h"
#include "stdlib.h"
#include "HostLoadFlux.h"
#include "GlobalVariablesHost.h"
#include "immintrin.h"
#define N_UNROLL 8
#include "newTimer.h"
using namespace newTimer;
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
	if (loopID == 0) printf("nBoundFace = %d, localEnd = %d\n", nBoundFace, localEnd);
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
	TIMERCPU0("HostFaceLoopLoadFluxCommonInside", "Common Face Loop with no Optimization");
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
	TIMERCPU1("HostFaceLoopLoadFluxCommonInside");

	resetRes(loopID);
	TIMERCPU0("HostFaceLoopLoadFluxCommonOutside", "Common Face Loop with no Optimization");
	//Faces on boundary faces
	for (equationID = 0; equationID < nEquation; equationID++){
		for (faceID = localStart; faceID < nMid; faceID++){
			le = leftCellofFace[faceID];
			res[equationID][le] -= flux[equationID][faceID];
		}
	}
	//Faces on interior faces
	for (equationID = 0; equationID < nEquation; equationID++){
		for (faceID = nMid; faceID < localEnd; faceID++){
			le = leftCellofFace[faceID];
			re = rightCellofFace[faceID];
			res[equationID][le] -= flux[equationID][faceID];
			res[equationID][re] += flux[equationID][faceID];
		}
	}
	TIMERCPU1("HostFaceLoopLoadFluxCommonOutside");
}
void HostCellLoopLoadFlux(const int loopID){
	if (loopID == 0) printf("HostCellLoopLoadFlux\n");
	int cellID, equationID;
	int faceID, numFacesInCell, offset, leftRight;
	int nEquation = nchem + nl;
	//reser Res by ResOrg
	resetRes(loopID);
	TIMERCPU0("HostCellLoopLoadCommonInside", "Cell Loop Common inside");
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
	TIMERCPU1("HostCellLoopLoadCommonInside");
	resetRes(loopID);
	TIMERCPU0("HostCellLoopLoadCommonOutside", "Cell Loop Common outside");
	for (equationID = 0; equationID < nEquation; equationID++){
		for (cellID = 0; cellID < nTotalCell; cellID++){
			numFacesInCell = faceNumberOfEachCell[cellID];
			for (offset = 0; offset < numFacesInCell; offset++){
				faceID = cell2Face[cellID][offset];
				leftRight = leftCellofFace[faceID];
				if (leftRight == cellID) res[equationID][cellID] -= flux[equationID][faceID];
				else  res[equationID][cellID] += flux[equationID][faceID]; 
			}
		}
	}
	TIMERCPU1("HostCellLoopLoadCommonOutside");
}
