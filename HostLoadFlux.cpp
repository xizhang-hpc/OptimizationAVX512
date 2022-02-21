#include "stdio.h"
#include "stdlib.h"
#include "HostLoadFlux.h"
#include "GlobalVariablesHost.h"
#include "immintrin.h"
#define N_UNROLL 8
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
void resetResAVX(const int loopID){
	if (loopID == 0) printf("resetResAVX\n");
	int equationID, cellID;
	int nEquation = nl + nchem;
	int nTotal = nTotalCell + nBoundFace;
	for (cellID = 0; cellID < nTotal; cellID++){
		for (equationID = 0; equationID < nEquation; equationID++){
			resAVX[equationID][cellID] = resOrg[equationID][cellID];
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
	int n_block = ((nMid - localStart)/N_UNROLL)*N_UNROLL;
	int n_tail = (nMid - localStart) - n_block;
	printf("localStart = %d, nMid = %d, n_block = %d, n_tail = %d\n", localStart, nMid, n_block, n_tail);
	for (faceID = localStart; faceID < n_block; faceID+=N_UNROLL){
		indexLeft = _mm256_maskz_loadu_epi32(0xFF, leftCellofFace + faceID);
		//indexLeft = _mm256_loadu_epi32(leftCellofFace + faceID);
		zmmFlux = _mm512_loadu_pd(flux[0]+faceID);
		zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[0], 8);
		zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
		_mm512_i32scatter_pd(resAVX[0], indexLeft, zmmSub, 8);
	}
	if (n_tail >= 0){
		__m512d zmmZero = _mm512_setzero_pd();	
		//__m256i indexZero = _mm256_maskz_set1_epi32(0xFF, 0);
		//indexLeft = _mm256_mask_loadu_epi32(indexZero, 0xFF>>(8-n_tail), leftCellofFace+n_block);
		indexLeft = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), leftCellofFace+n_block);
		zmmFlux = _mm512_maskz_loadu_pd(0xFF>>(8-n_tail), flux[0]+n_block);
		zmmRes = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexLeft, resAVX[0], 8);
		zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
		_mm512_mask_i32scatter_pd(resAVX[0], 0xFF>>(8-n_tail), indexLeft, zmmSub, 8);
	}
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
	resetResAVX(loopID);
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
	//AVX512
	//obtain posiCell2Face
	int * posiCell2Face = (int *)malloc(nTotalCell * sizeof(int));
	int * cellIDArray = (int *)malloc(nTotalCell * sizeof(int));
	posiCell2Face[0] = 0;
	cellIDArray[0] = 0;
	for (cellID = 1; cellID < nTotalCell; cellID++){
		posiCell2Face[cellID] = posiCell2Face[cellID-1] + faceNumberOfEachCell[cellID-1];
		cellIDArray[cellID] = cellID;
	}
	int maxFaceNum = 6;
	__m256i zmmNumFace;
	__m256i zmmPosiCell;
	__m256i zmmCellID;
	__m256i zmmFaceID;
	__m256i zmmLeftRight;
	__m256i zmmOffset;
	__m256i zmmOffReal;
	__m512d zmmRes;
	__m512d zmmFlux;
	__m512d zmmPositive = _mm512_set1_pd(1.0);
	__m512d zmmNegative = _mm512_set1_pd(-1.0);
	__m512d zmmPN;
	__m512d zmmFluxTmp;
	__m512d zmm0 = _mm512_setzero_pd();
	__mmask kCellID;
	__mmask kZero;
	__m256i zmmZero = _mm256_maskz_set1_epi32(0xFF, 0);
	__m256i zmmNum;
/*
	__m256i zmmOne = _mm256_maskz_set1_epi32(0xFF, 1);
	__m256i zmmTwo = _mm256_maskz_set1_epi32(0xFF, 2);
	__m256i zmmThree = _mm256_maskz_set1_epi32(0xFF, 3);
	__m256i zmmFour = _mm256_maskz_set1_epi32(0xFF, 4);
	__m256i zmmFive = _mm256_maskz_set1_epi32(0xFF, 5);
	__m256i zmmSix = _mm256_maskz_set1_epi32(0xFF, 6);
*/
	__m256i zmmMinusOne = _mm256_maskz_set1_epi32(0xFF, -1);
	__m256i zmmPosiReal;
	int n_block = (nTotalCell/N_UNROLL)*N_UNROLL;
	int n_tail = nTotalCell - n_block;
	printf("nTotalCell = %d, n_block = %d, n_tail = %d\n", nTotalCell, n_block, n_tail);
	for (cellID = 0; cellID < n_block; cellID += N_UNROLL){
		zmmRes = _mm512_loadu_pd(resAVX[0]+cellID);
		zmmNumFace = _mm256_maskz_loadu_epi32(0xFF, faceNumberOfEachCell + cellID);
		zmmCellID = _mm256_maskz_loadu_epi32(0xFF, cellIDArray + cellID);
		zmmPosiCell = _mm256_maskz_loadu_epi32(0xFF, posiCell2Face + cellID);
	    for (int offsetFace =1; offsetFace <= maxFaceNum; offsetFace++){	
		zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetFace);
		zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNumFace, zmmNum);
		kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
                zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmZero, zmmOffset);
		zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmPosiCell, zmmOffReal);

		zmmFaceID = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmPosiReal, cell2Face[0], 4);
		zmmLeftRight = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, leftCellofFace, 4);
		zmmFlux = _mm512_i32gather_pd(zmmFaceID, flux[0], 8);
		kCellID = _mm256_cmp_epi32_mask(zmmLeftRight, zmmCellID, _MM_CMPINT_EQ);
		zmmPN = _mm512_mask_blend_pd(kCellID, zmmPositive, zmmNegative);
		zmmFluxTmp = _mm512_maskz_mul_pd(kZero, zmmPN, zmmFlux);
		zmmRes = _mm512_add_pd(zmmRes, zmmFluxTmp);
            }
        	_mm512_storeu_pd(resAVX[0]+cellID, zmmRes);

	}
	//for n_tail
	if (n_tail >= 0){
		__mmask8 kNTail = 0xFF>>(8-n_tail);
		zmmRes = _mm512_maskz_loadu_pd(kNTail ,resAVX[0]+n_block);
		zmmNumFace = _mm256_maskz_loadu_epi32(kNTail, faceNumberOfEachCell + n_block);
		zmmCellID = _mm256_maskz_loadu_epi32(kNTail, cellIDArray + n_block);
		zmmPosiCell = _mm256_maskz_loadu_epi32(kNTail, posiCell2Face + n_block);
	    for (int offsetFace =1; offsetFace <= maxFaceNum; offsetFace++){	
		zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetFace);
		zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNumFace, zmmNum);
		kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
                zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmZero, zmmOffset);
		zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmPosiCell, zmmOffReal);

		zmmFaceID = _mm256_mmask_i32gather_epi32(zmmZero, kNTail, zmmPosiReal, cell2Face[0], 4);
		zmmLeftRight = _mm256_mmask_i32gather_epi32(zmmZero, kNTail, zmmFaceID, leftCellofFace, 4);
		zmmFlux = _mm512_mask_i32gather_pd(zmm0, kNTail, zmmFaceID, flux[0], 8);
		kCellID = _mm256_cmp_epi32_mask(zmmLeftRight, zmmCellID, _MM_CMPINT_EQ);
		zmmPN = _mm512_mask_blend_pd(kCellID, zmmPositive, zmmNegative);
		zmmFluxTmp = _mm512_maskz_mul_pd(kZero, zmmPN, zmmFlux);
		zmmRes = _mm512_add_pd(zmmRes, zmmFluxTmp);
            }
		_mm512_mask_compressstoreu_pd(resAVX[0]+n_block, kNTail, zmmRes);

	}
	
	//valiate resAVX[0]
	for (int cellID = 0; cellID < nTotalCell; cellID++){
		if (abs(res[0][cellID] - resAVX[0][cellID]) > 1.0e-8) {
			printf("on cellID %d, res %.30e is not equal to resAVX %.30e\n", cellID, res[0][cellID], resAVX[0][cellID]);
			exit(0);
		}
	}
	printf("Validate successfully!\n");
}
