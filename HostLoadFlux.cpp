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
	//for AVX512
	__m512d zmmFlux;
	__m512d zmmRes;
	__m512d zmmSub;
	__m256i indexLeft;
	__m256i indexColor;
	__m256i indexZero = _mm256_maskz_set1_epi32(0xFF, 0);
    //for (equationID = 0; equationID < nEquation; equationID++){
	for (int colorID = 0; colorID < BoundFaceColorNum; colorID++){
		int colorGroupNum = BoundFaceGroupNum[colorID];
		int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
		int n_tail = colorGroupNum - n_block;
		int colorGroupStart = BoundFaceGroupPosi[colorID];
		printf("colorID = %d, localStart = %d, nMid = %d, n_block = %d, n_tail = %d\n", colorID, localStart, nMid, n_block, n_tail);
		for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
			indexColor = _mm256_maskz_loadu_epi32(0xFF, BoundFaceGroup + colorGroupStart + colorGroupID);
			//indexLeft = _mm256_maskz_loadu_epi32(0xFF, leftCellofFace + faceID);
			indexLeft = _mm256_mmask_i32gather_epi32(indexZero, 0xFF, indexColor, leftCellofFace, 4);;
			//zmmFlux = _mm512_loadu_pd(flux[0]+faceID);
			zmmFlux = _mm512_i32gather_pd(indexColor, flux[0], 8);
			zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[0], 8);
			zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
			_mm512_i32scatter_pd(resAVX[0], indexLeft, zmmSub, 8);
		}
		if (n_tail >= 0){
			__m512d zmmZero = _mm512_setzero_pd();
			indexColor = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), BoundFaceGroup + colorGroupStart + n_block);	
		//__m256i indexZero = _mm256_maskz_set1_epi32(0xFF, 0);
		//indexLeft = _mm256_mask_loadu_epi32(indexZero, 0xFF>>(8-n_tail), leftCellofFace+n_block);
		//indexLeft = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), leftCellofFace+n_block);
			indexLeft = _mm256_mmask_i32gather_epi32(indexZero, 0xFF>>(8-n_tail), indexColor, leftCellofFace, 4);;
		//zmmFlux = _mm512_maskz_loadu_pd(0xFF>>(8-n_tail), flux[0]+n_block);
			zmmFlux = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexColor, flux[0], 8);
			zmmRes = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexLeft, resAVX[0], 8);
			zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
			_mm512_mask_i32scatter_pd(resAVX[0], 0xFF>>(8-n_tail), indexLeft, zmmSub, 8);
		} //end if
	 } //end colorID loop
    //}//end equationID loop
	//validate It shows that face coloring is required!
	for (faceID = localStart; faceID < nMid; faceID++){
		le = leftCellofFace[faceID];
			if (res[0][le] -  resAVX[0][le]) {
				printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[0][le], resAVX[0][le]);
				printf("nBoundFace = %d, on cell %d , onws faces:\n", nBoundFace, le);
				int numFaceInCell = faceNumberOfEachCell[le];
				for (int offset = 0; offset < numFaceInCell; offset++){
					printf("%d\n", cell2Face[le][offset]);
				}
				
			}
	}
	printf("validate boundary successfully\n");
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
