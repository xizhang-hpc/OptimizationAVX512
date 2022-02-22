#include "stdio.h"
#include "stdlib.h"
#include "AVX512LoadFlux.h"
#include "GlobalVariablesHost.h"
#include "immintrin.h"
#define N_UNROLL 8
#include "newTimer.h"
using namespace newTimer;
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

void AVX512CellLoopLoadFluxOutside(int loopID){
	if (loopID == 0) printf("AVX512CellLoopLoadFluxOutside\n");
	int cellID, equationID;
	int faceID, numFacesInCell, offset, leftRight;
	int nEquation = nchem + nl;

	int * cellIDArray = (int *)malloc(nTotalCell * sizeof(int));
	cellIDArray[0] = 0;
	for (cellID = 1; cellID < nTotalCell; cellID++){
		cellIDArray[cellID] = cellID;
	}
	int maxFaceNum = 6;
	//reser Res by ResOrg
	resetResAVX(loopID);
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
	__m256i zmmMinusOne = _mm256_maskz_set1_epi32(0xFF, -1);
	__m256i zmmPosiReal;
	int n_block = (nTotalCell/N_UNROLL)*N_UNROLL;
	int n_tail = nTotalCell - n_block;
	//printf("nTotalCell = %d, n_block = %d, n_tail = %d\n", nTotalCell, n_block, n_tail);
TIMERCPU0("HostCellLoopLoadAVXOutside", "Cell Loop Common outside");
     for (equationID = 0; equationID < nEquation; equationID++){
	for (cellID = 0; cellID < n_block; cellID += N_UNROLL){
		zmmRes = _mm512_loadu_pd(resAVX[equationID]+cellID);
		zmmNumFace = _mm256_maskz_loadu_epi32(0xFF, faceNumberOfEachCell + cellID);
		zmmCellID = _mm256_maskz_loadu_epi32(0xFF, cellIDArray + cellID);
		zmmPosiCell = _mm256_maskz_loadu_epi32(0xFF, cell2FacePosition + cellID);
	    for (int offsetFace =1; offsetFace <= maxFaceNum; offsetFace++){	
		zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetFace);
		zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNumFace, zmmNum);
		kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
                zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmZero, zmmOffset);
		zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmPosiCell, zmmOffReal);

		zmmFaceID = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmPosiReal, cell2Face[0], 4);
		zmmLeftRight = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, leftCellofFace, 4);
		zmmFlux = _mm512_i32gather_pd(zmmFaceID, flux[equationID], 8);
		kCellID = _mm256_cmp_epi32_mask(zmmLeftRight, zmmCellID, _MM_CMPINT_EQ);
		zmmPN = _mm512_mask_blend_pd(kCellID, zmmPositive, zmmNegative);
		zmmFluxTmp = _mm512_maskz_mul_pd(kZero, zmmPN, zmmFlux);
		zmmRes = _mm512_add_pd(zmmRes, zmmFluxTmp);
            }
        	_mm512_storeu_pd(resAVX[equationID]+cellID, zmmRes);
	}
	//for n_tail
	if (n_tail >= 0){
		__mmask8 kNTail = 0xFF>>(8-n_tail);
		zmmRes = _mm512_maskz_loadu_pd(kNTail ,resAVX[equationID]+n_block);
		zmmNumFace = _mm256_maskz_loadu_epi32(kNTail, faceNumberOfEachCell + n_block);
		zmmCellID = _mm256_maskz_loadu_epi32(kNTail, cellIDArray + n_block);
		zmmPosiCell = _mm256_maskz_loadu_epi32(kNTail, cell2FacePosition + n_block);
	    for (int offsetFace =1; offsetFace <= maxFaceNum; offsetFace++){	
		zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetFace);
		zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNumFace, zmmNum);
		kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
                zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmZero, zmmOffset);
		zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmPosiCell, zmmOffReal);

		zmmFaceID = _mm256_mmask_i32gather_epi32(zmmZero, kNTail, zmmPosiReal, cell2Face[0], 4);
		zmmLeftRight = _mm256_mmask_i32gather_epi32(zmmZero, kNTail, zmmFaceID, leftCellofFace, 4);
		zmmFlux = _mm512_mask_i32gather_pd(zmm0, kNTail, zmmFaceID, flux[equationID], 8);
		kCellID = _mm256_cmp_epi32_mask(zmmLeftRight, zmmCellID, _MM_CMPINT_EQ);
		zmmPN = _mm512_mask_blend_pd(kCellID, zmmPositive, zmmNegative);
		zmmFluxTmp = _mm512_maskz_mul_pd(kZero, zmmPN, zmmFlux);
		zmmRes = _mm512_add_pd(zmmRes, zmmFluxTmp);
            }
		_mm512_mask_compressstoreu_pd(resAVX[equationID]+n_block, kNTail, zmmRes);

	}
   }	
TIMERCPU1("HostCellLoopLoadAVXOutside");
	//valiate resAVX[0]
	if (loopID == 0) {
	     for (equationID = 0; equationID < nEquation; equationID++){
		for (int cellID = 0; cellID < nTotalCell; cellID++){
			if (abs(res[equationID][cellID] - resAVX[equationID][cellID]) > 1.0e-8) {
				printf("on equationID %d on cellID %d, res %.30e is not equal to resAVX %.30e\n", equationID, cellID, res[equationID][cellID], resAVX[equationID][cellID]);
				exit(0);
			}
		}
     	     }
	     printf("Validate AVX512 Outside successfully!\n");
	}

}

void AVX512CellLoopLoadFluxInside(int loopID){
	if (loopID == 0) printf("AVX512CellLoopLoadFluxInside\n");
	int cellID, equationID;
	int faceID, numFacesInCell, offset, leftRight;
	int nEquation = nchem + nl;

	int * cellIDArray = (int *)malloc(nTotalCell * sizeof(int));
	cellIDArray[0] = 0;
	for (cellID = 1; cellID < nTotalCell; cellID++){
		cellIDArray[cellID] = cellID;
	}
	int maxFaceNum = 6;
	//reser Res by ResOrg
	resetResAVX(loopID);
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
	__m256i zmmMinusOne = _mm256_maskz_set1_epi32(0xFF, -1);
	__m256i zmmPosiReal;
	int n_block = (nTotalCell/N_UNROLL)*N_UNROLL;
	int n_tail = nTotalCell - n_block;
	//printf("nTotalCell = %d, n_block = %d, n_tail = %d\n", nTotalCell, n_block, n_tail);
TIMERCPU0("HostCellLoopLoadAVXInside", "Cell Loop Common inside");
	for (cellID = 0; cellID < n_block; cellID += N_UNROLL){
		zmmNumFace = _mm256_maskz_loadu_epi32(0xFF, faceNumberOfEachCell + cellID);
		zmmCellID = _mm256_maskz_loadu_epi32(0xFF, cellIDArray + cellID);
		zmmPosiCell = _mm256_maskz_loadu_epi32(0xFF, cell2FacePosition + cellID);
          for (equationID = 0; equationID < nEquation; equationID++){
	    zmmRes = _mm512_loadu_pd(resAVX[equationID]+cellID);
	    for (int offsetFace =1; offsetFace <= maxFaceNum; offsetFace++){	
		zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetFace);
		zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNumFace, zmmNum);
		kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
                zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmZero, zmmOffset);
		zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmPosiCell, zmmOffReal);

		zmmFaceID = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmPosiReal, cell2Face[0], 4);
		zmmLeftRight = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, leftCellofFace, 4);
		zmmFlux = _mm512_i32gather_pd(zmmFaceID, flux[equationID], 8);
		kCellID = _mm256_cmp_epi32_mask(zmmLeftRight, zmmCellID, _MM_CMPINT_EQ);
		zmmPN = _mm512_mask_blend_pd(kCellID, zmmPositive, zmmNegative);
		zmmFluxTmp = _mm512_maskz_mul_pd(kZero, zmmPN, zmmFlux);
		zmmRes = _mm512_add_pd(zmmRes, zmmFluxTmp);
            }
        	_mm512_storeu_pd(resAVX[equationID]+cellID, zmmRes);
	  }
	}
	//for n_tail
	if (n_tail >= 0){
		__mmask8 kNTail = 0xFF>>(8-n_tail);
		zmmNumFace = _mm256_maskz_loadu_epi32(kNTail, faceNumberOfEachCell + n_block);
		zmmCellID = _mm256_maskz_loadu_epi32(kNTail, cellIDArray + n_block);
		zmmPosiCell = _mm256_maskz_loadu_epi32(kNTail, cell2FacePosition + n_block);
          for (equationID = 0; equationID < nEquation; equationID++){
	    zmmRes = _mm512_maskz_loadu_pd(kNTail ,resAVX[equationID]+n_block);
	    for (int offsetFace =1; offsetFace <= maxFaceNum; offsetFace++){	
		zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetFace);
		zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNumFace, zmmNum);
		kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
                zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmZero, zmmOffset);
		zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmPosiCell, zmmOffReal);

		zmmFaceID = _mm256_mmask_i32gather_epi32(zmmZero, kNTail, zmmPosiReal, cell2Face[0], 4);
		zmmLeftRight = _mm256_mmask_i32gather_epi32(zmmZero, kNTail, zmmFaceID, leftCellofFace, 4);
		zmmFlux = _mm512_mask_i32gather_pd(zmm0, kNTail, zmmFaceID, flux[equationID], 8);
		kCellID = _mm256_cmp_epi32_mask(zmmLeftRight, zmmCellID, _MM_CMPINT_EQ);
		zmmPN = _mm512_mask_blend_pd(kCellID, zmmPositive, zmmNegative);
		zmmFluxTmp = _mm512_maskz_mul_pd(kZero, zmmPN, zmmFlux);
		zmmRes = _mm512_add_pd(zmmRes, zmmFluxTmp);
            }
	    _mm512_mask_compressstoreu_pd(resAVX[equationID]+n_block, kNTail, zmmRes);
          }

	}
TIMERCPU1("HostCellLoopLoadAVXInside");
	//valiate resAVX[0]
	if (loopID == 0) {
	     for (equationID = 0; equationID < nEquation; equationID++){
		for (int cellID = 0; cellID < nTotalCell; cellID++){
			if (abs(res[equationID][cellID] - resAVX[equationID][cellID]) > 1.0e-8) {
				printf("on equationID %d on cellID %d, res %.30e is not equal to resAVX %.30e\n", equationID, cellID, res[equationID][cellID], resAVX[equationID][cellID]);
				exit(0);
			}
		}
     	     }
	     printf("Validate AVX512 Inside successfully!\n");
	}

}
