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
TIMERCPU0("HostCellLoopLoadAVXOutside", "Cell Loop AVX512 outside");
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
TIMERCPU0("HostCellLoopLoadAVXInside", "Cell Loop AVX512 inside");
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

void AVX512FaceLoopLoadFluxOutside(int loopID){
	if (loopID == 0) printf("AVX512FaceLoopLoadFluxOutside\n");

	int faceID, equationID;
	int le, re;
	int nEquation = nl + nchem;

	resetResAVX(loopID);

	__m512d zmmFlux;
	__m512d zmmRes;
	__m512d zmmSub;
	__m512d zmmAdd;
	__m256i indexLeft;
	__m256i indexRight;
	__m256i indexColor;
	__m256i indexZero = _mm256_maskz_set1_epi32(0xFF, 0);
	//boundary faces
	TIMERCPU0("HostFaceLoopLoadAVXOutside", "Face Loop AVX512 outside");
    	for (equationID = 0; equationID < nEquation; equationID++){
		for (int colorID = 0; colorID < BoundFaceColorNum; colorID++){
			int colorGroupNum = BoundFaceGroupNum[colorID];
			int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
			int n_tail = colorGroupNum - n_block;
			int colorGroupStart = BoundFaceGroupPosi[colorID];
			//printf("colorID = %d, colorGroupNum = %d, n_block = %d, n_tail = %d\n", colorID, colorGroupNum, n_block, n_tail);
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				indexColor = _mm256_maskz_loadu_epi32(0xFF, BoundFaceGroup + colorGroupStart + colorGroupID);
				indexLeft = _mm256_mmask_i32gather_epi32(indexZero, 0xFF, indexColor, leftCellofFace, 4);;
				zmmFlux = _mm512_i32gather_pd(indexColor, flux[equationID], 8);
				zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexLeft, zmmSub, 8);
			}
			if (n_tail >= 0){
				__m512d zmmZero = _mm512_setzero_pd();
				indexColor = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), BoundFaceGroup + colorGroupStart + n_block);	
				indexLeft = _mm256_mmask_i32gather_epi32(indexZero, 0xFF>>(8-n_tail), indexColor, leftCellofFace, 4);;
				zmmFlux = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexColor, flux[equationID], 8);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], 0xFF>>(8-n_tail), indexLeft, zmmSub, 8);
			} //end if
		} //end colorID loop
	}//end equationID loop
    
	//Interior faces
	for (equationID = 0; equationID < nEquation; equationID++){
		for (int colorID = 0; colorID < InteriorFaceColorNum; colorID++){
			int colorGroupNum = InteriorFaceGroupNum[colorID];
			int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
			int n_tail = colorGroupNum - n_block;
			int colorGroupStart = InteriorFaceGroupPosi[colorID];
			//printf("colorID = %d, colorGroupNum = %d, n_block = %d, n_tail = %d\n", colorID, colorGroupNum, n_block, n_tail);
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				indexColor = _mm256_maskz_loadu_epi32(0xFF, InteriorFaceGroup + colorGroupStart + colorGroupID);
				indexLeft = _mm256_mmask_i32gather_epi32(indexZero, 0xFF, indexColor, leftCellofFace, 4);
				indexRight = _mm256_mmask_i32gather_epi32(indexZero, 0xFF, indexColor, rightCellofFace, 4);
				zmmFlux = _mm512_i32gather_pd(indexColor, flux[equationID], 8);
				zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexLeft, zmmSub, 8);
				zmmRes = _mm512_i32gather_pd(indexRight, resAVX[equationID], 8);
				zmmAdd = _mm512_add_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexRight, zmmAdd, 8);

			}
			//for n_tail
			if (n_tail > 0){
				__mmask8 kNTail = 0xFF>>(8-n_tail);
				__m512d zmmZero = _mm512_setzero_pd();
				indexColor = _mm256_maskz_loadu_epi32(kNTail, InteriorFaceGroup + colorGroupStart + n_block);	
				indexLeft = _mm256_mmask_i32gather_epi32(indexZero, kNTail, indexColor, leftCellofFace, 4);
				indexRight = _mm256_mmask_i32gather_epi32(indexZero, kNTail, indexColor, rightCellofFace, 4);
				zmmFlux = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexColor, flux[equationID], 8);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], kNTail, indexLeft, zmmSub, 8);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexRight, resAVX[equationID], 8);
				zmmAdd = _mm512_add_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], kNTail, indexRight, zmmAdd, 8);
			}//end n_tail
		}// end color loop
	}
	TIMERCPU1("HostFaceLoopLoadAVXOutside");
	//validate
	if (loopID == 0){
		for (equationID = 0; equationID < nEquation; equationID++){
			for (faceID = 0; faceID < nBoundFace; faceID++){
				le = leftCellofFace[faceID];
				if (abs(res[equationID][le]-resAVX[equationID][le])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[equationID][le], resAVX[equationID][le]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[le][offset]);
					}
					exit(0);
				}
			}
			for (faceID = nBoundFace; faceID < nTotalFace; faceID++){
				le = leftCellofFace[faceID];
				re = rightCellofFace[faceID];
				if (abs(res[equationID][le]-resAVX[equationID][le])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[equationID][le], resAVX[equationID][le]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[le][offset]);
					}
					exit(0);
				}
				if (abs(res[equationID][re]-resAVX[equationID][re])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, re, res[equationID][re], resAVX[equationID][re]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[re][offset]);
					}
					exit(0);
				}
			}
    		}
		printf("validate AVX512 faceLoadFlux outside successfully\n");
	}
}

void AVX512FaceLoopLoadFluxInside(int loopID){
	if (loopID == 0) printf("AVX512FaceLoopLoadFluxInside\n");

	int faceID, equationID;
	int le, re;
	int nEquation = nl + nchem;

	resetResAVX(loopID);

	__m512d zmmFlux;
	__m512d zmmRes;
	__m512d zmmSub;
	__m512d zmmAdd;
	__m256i indexLeft;
	__m256i indexRight;
	__m256i indexColor;
	__m256i indexZero = _mm256_maskz_set1_epi32(0xFF, 0);
	//boundary faces
	TIMERCPU0("HostFaceLoopLoadAVXInside", "Face Loop AVX512 inside");
	for (int colorID = 0; colorID < BoundFaceColorNum; colorID++){
		int colorGroupNum = BoundFaceGroupNum[colorID];
		int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
		int n_tail = colorGroupNum - n_block;
		int colorGroupStart = BoundFaceGroupPosi[colorID];
    		for (equationID = 0; equationID < nEquation; equationID++){
			//printf("colorID = %d, colorGroupNum = %d, n_block = %d, n_tail = %d\n", colorID, colorGroupNum, n_block, n_tail);
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				indexColor = _mm256_maskz_loadu_epi32(0xFF, BoundFaceGroup + colorGroupStart + colorGroupID);
				indexLeft = _mm256_mmask_i32gather_epi32(indexZero, 0xFF, indexColor, leftCellofFace, 4);;
				zmmFlux = _mm512_i32gather_pd(indexColor, flux[equationID], 8);
				zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexLeft, zmmSub, 8);
			}
			if (n_tail >= 0){
				__m512d zmmZero = _mm512_setzero_pd();
				indexColor = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), BoundFaceGroup + colorGroupStart + n_block);	
				indexLeft = _mm256_mmask_i32gather_epi32(indexZero, 0xFF>>(8-n_tail), indexColor, leftCellofFace, 4);;
				zmmFlux = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexColor, flux[equationID], 8);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], 0xFF>>(8-n_tail), indexLeft, zmmSub, 8);
			} //end if
		} //end colorID loop
	}//end equationID loop
    
	//Interior faces
	for (int colorID = 0; colorID < InteriorFaceColorNum; colorID++){
		int colorGroupNum = InteriorFaceGroupNum[colorID];
		int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
		int n_tail = colorGroupNum - n_block;
		//for (int faceID = faceID; faceID < n_block; faceID+=N_UNROLL){
		int colorGroupStart = InteriorFaceGroupPosi[colorID];
		for (equationID = 0; equationID < nEquation; equationID++){
			//printf("colorID = %d, colorGroupNum = %d, n_block = %d, n_tail = %d\n", colorID, colorGroupNum, n_block, n_tail);
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				indexColor = _mm256_maskz_loadu_epi32(0xFF, InteriorFaceGroup + colorGroupStart + colorGroupID);
				indexLeft = _mm256_mmask_i32gather_epi32(indexZero, 0xFF, indexColor, leftCellofFace, 4);
				indexRight = _mm256_mmask_i32gather_epi32(indexZero, 0xFF, indexColor, rightCellofFace, 4);
				zmmFlux = _mm512_i32gather_pd(indexColor, flux[equationID], 8);
				zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexLeft, zmmSub, 8);
				zmmRes = _mm512_i32gather_pd(indexRight, resAVX[equationID], 8);
				zmmAdd = _mm512_add_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexRight, zmmAdd, 8);

			}
			//for n_tail
			if (n_tail > 0){
				__mmask8 kNTail = 0xFF>>(8-n_tail);
				__m512d zmmZero = _mm512_setzero_pd();
				indexColor = _mm256_maskz_loadu_epi32(kNTail, InteriorFaceGroup + colorGroupStart + n_block);	
				indexLeft = _mm256_mmask_i32gather_epi32(indexZero, kNTail, indexColor, leftCellofFace, 4);
				indexRight = _mm256_mmask_i32gather_epi32(indexZero, kNTail, indexColor, rightCellofFace, 4);
				zmmFlux = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexColor, flux[equationID], 8);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], kNTail, indexLeft, zmmSub, 8);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexRight, resAVX[equationID], 8);
				zmmAdd = _mm512_add_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], kNTail, indexRight, zmmAdd, 8);
			}//end n_tail
		}// end equation loop
	}//end color loop
	TIMERCPU1("HostFaceLoopLoadAVXInside");
	//validate
	if (loopID == 0){
		for (equationID = 0; equationID < nEquation; equationID++){
			for (faceID = 0; faceID < nBoundFace; faceID++){
				le = leftCellofFace[faceID];
				if (abs(res[equationID][le]-resAVX[equationID][le])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[equationID][le], resAVX[equationID][le]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[le][offset]);
					}
					exit(0);
				}
			}
			for (faceID = nBoundFace; faceID < nTotalFace; faceID++){
				le = leftCellofFace[faceID];
				re = rightCellofFace[faceID];
				if (abs(res[equationID][le]-resAVX[equationID][le])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[equationID][le], resAVX[equationID][le]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[le][offset]);
					}
					exit(0);
				}
				if (abs(res[equationID][re]-resAVX[equationID][re])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, re, res[equationID][re], resAVX[equationID][re]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[re][offset]);
					}
					exit(0);
				}
			}
    		}
		printf("validate AVX512 faceLoadFlux inside successfully\n");
	}
}

void AVX512FaceLoopLoadFluxInsideReOpt(int loopID){
	if (loopID == 0) printf("AVX512FaceLoopLoadFluxInsideReOpt\n");

	int faceID, equationID;
	int le, re;
	int nEquation = nl + nchem;

	resetResAVX(loopID);

	__m512d zmmFlux;
	__m512d zmmRes;
	__m512d zmmSub;
	__m512d zmmAdd;
	__m256i indexLeft;
	__m256i indexRight;
	//__m256i indexColor;
	__m256i indexZero = _mm256_maskz_set1_epi32(0xFF, 0);
	//boundary faces
	TIMERCPU0("HostFaceLoopLoadAVXInsideReOpt", "Face Loop AVX512 inside re");
	for (int colorID = 0; colorID < BoundFaceColorNum; colorID++){
		int colorGroupNum = BoundFaceGroupNum[colorID];
		int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
		int n_tail = colorGroupNum - n_block;
		int colorGroupStart = BoundFaceGroupPosi[colorID];
    		for (equationID = 0; equationID < nEquation; equationID++){
			//printf("colorID = %d, colorGroupNum = %d, n_block = %d, n_tail = %d\n", colorID, colorGroupNum, n_block, n_tail);
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				indexLeft = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceRe + colorGroupStart + colorGroupID);
				zmmFlux = _mm512_loadu_pd(fluxRe[equationID]+ colorGroupStart + colorGroupID);
				zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexLeft, zmmSub, 8);
			}
			if (n_tail >= 0){
				__m512d zmmZero = _mm512_setzero_pd();
				indexLeft = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), leftCellofFaceRe + colorGroupStart + n_block);
				zmmFlux = _mm512_maskz_loadu_pd(0xFF>>(8-n_tail), fluxRe[equationID]+colorGroupStart+n_block);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], 0xFF>>(8-n_tail), indexLeft, zmmSub, 8);
			} //end if
		} //end equationID loop
	}//end colorID loop
    
	//Interior faces
	for (int colorID = 0; colorID < InteriorFaceColorNum; colorID++){
		int colorGroupNum = InteriorFaceGroupNum[colorID];
		int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
		int n_tail = colorGroupNum - n_block;
		int colorGroupStart = InteriorFaceGroupPosi[colorID] + nBoundFace;
		//printf("colorID = %d, colorGroupNum = %d, n_block = %d, n_tail = %d\n", colorID, colorGroupNum, n_block, n_tail);
		for (equationID = 0; equationID < nEquation; equationID++){
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				indexLeft = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceRe + colorGroupStart + colorGroupID);
				indexRight = _mm256_maskz_loadu_epi32(0xFF, rightCellofFaceRe + colorGroupStart + colorGroupID);
				zmmFlux = _mm512_loadu_pd(fluxRe[equationID]+ colorGroupStart + colorGroupID);
				zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexLeft, zmmSub, 8);
				zmmRes = _mm512_i32gather_pd(indexRight, resAVX[equationID], 8);
				zmmAdd = _mm512_add_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexRight, zmmAdd, 8);

			}
			//for n_tail
			if (n_tail > 0){
				__mmask8 kNTail = 0xFF>>(8-n_tail);
				__m512d zmmZero = _mm512_setzero_pd();
				indexLeft = _mm256_maskz_loadu_epi32(kNTail, leftCellofFaceRe + colorGroupStart + n_block);
				indexRight = _mm256_maskz_loadu_epi32(kNTail, rightCellofFaceRe + colorGroupStart + n_block);
				zmmFlux = _mm512_maskz_loadu_pd(kNTail, fluxRe[equationID]+colorGroupStart+n_block);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], kNTail, indexLeft, zmmSub, 8);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexRight, resAVX[equationID], 8);
				zmmAdd = _mm512_add_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], kNTail, indexRight, zmmAdd, 8);
			}//end n_tail
		}// end color loop
	}
	TIMERCPU1("HostFaceLoopLoadAVXInsideReOpt");
	//validate
	if (loopID == 0){
		for (equationID = 0; equationID < nEquation; equationID++){
			for (faceID = 0; faceID < nBoundFace; faceID++){
				le = leftCellofFace[faceID];
				if (abs(res[equationID][le]-resAVX[equationID][le])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[equationID][le], resAVX[equationID][le]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[le][offset]);
					}
					exit(0);
				}
			}
			for (faceID = nBoundFace; faceID < nTotalFace; faceID++){
				le = leftCellofFace[faceID];
				re = rightCellofFace[faceID];
				if (abs(res[equationID][le]-resAVX[equationID][le])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[equationID][le], resAVX[equationID][le]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[le][offset]);
					}
					exit(0);
				}
				if (abs(res[equationID][re]-resAVX[equationID][re])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, re, res[equationID][re], resAVX[equationID][re]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[re][offset]);
					}
					exit(0);
				}
			}
    		}
		printf("validate AVX512 faceLoadFlux inside opt successfully\n");
	}
}

void AVX512FaceLoopLoadFluxOutsideReOpt(int loopID){
	if (loopID == 0) printf("AVX512FaceLoopLoadFluxOutsideReOpt\n");

	int faceID, equationID;
	int le, re;
	int nEquation = nl + nchem;

	resetResAVX(loopID);

	__m512d zmmFlux;
	__m512d zmmRes;
	__m512d zmmSub;
	__m512d zmmAdd;
	__m256i indexLeft;
	__m256i indexRight;
	//__m256i indexColor;
	__m256i indexZero = _mm256_maskz_set1_epi32(0xFF, 0);
	//boundary faces
	TIMERCPU0("HostFaceLoopLoadAVXOutsideReOpt", "Face Loop AVX512 outside re");
    	for (equationID = 0; equationID < nEquation; equationID++){
		for (int colorID = 0; colorID < BoundFaceColorNum; colorID++){
			int colorGroupNum = BoundFaceGroupNum[colorID];
			int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
			int n_tail = colorGroupNum - n_block;
			int colorGroupStart = BoundFaceGroupPosi[colorID];
			//printf("colorID = %d, colorGroupNum = %d, n_block = %d, n_tail = %d\n", colorID, colorGroupNum, n_block, n_tail);
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				indexLeft = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceRe + colorGroupStart + colorGroupID);
				zmmFlux = _mm512_loadu_pd(fluxRe[equationID]+ colorGroupStart + colorGroupID);
				zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexLeft, zmmSub, 8);
			}
			if (n_tail >= 0){
				__m512d zmmZero = _mm512_setzero_pd();
				indexLeft = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), leftCellofFaceRe + colorGroupStart + n_block);
				zmmFlux = _mm512_maskz_loadu_pd(0xFF>>(8-n_tail), fluxRe[equationID]+colorGroupStart+n_block);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, 0xFF>>(8-n_tail), indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], 0xFF>>(8-n_tail), indexLeft, zmmSub, 8);
			} //end if
		} //end colorID loop
	}//end equationID loop
    
	//Interior faces
	for (equationID = 0; equationID < nEquation; equationID++){
		for (int colorID = 0; colorID < InteriorFaceColorNum; colorID++){
			int colorGroupNum = InteriorFaceGroupNum[colorID];
			int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
			int n_tail = colorGroupNum - n_block;
			int colorGroupStart = InteriorFaceGroupPosi[colorID] + nBoundFace;
			//printf("colorID = %d, colorGroupNum = %d, n_block = %d, n_tail = %d\n", colorID, colorGroupNum, n_block, n_tail);
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				indexLeft = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceRe + colorGroupStart + colorGroupID);
				indexRight = _mm256_maskz_loadu_epi32(0xFF, rightCellofFaceRe + colorGroupStart + colorGroupID);
				zmmFlux = _mm512_loadu_pd(fluxRe[equationID]+ colorGroupStart + colorGroupID);
				zmmRes = _mm512_i32gather_pd(indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexLeft, zmmSub, 8);
				zmmRes = _mm512_i32gather_pd(indexRight, resAVX[equationID], 8);
				zmmAdd = _mm512_add_pd(zmmRes, zmmFlux);
				_mm512_i32scatter_pd(resAVX[equationID], indexRight, zmmAdd, 8);

			}
			//for n_tail
			if (n_tail > 0){
				__mmask8 kNTail = 0xFF>>(8-n_tail);
				__m512d zmmZero = _mm512_setzero_pd();
				indexLeft = _mm256_maskz_loadu_epi32(kNTail, leftCellofFaceRe + colorGroupStart + n_block);
				indexRight = _mm256_maskz_loadu_epi32(kNTail, rightCellofFaceRe + colorGroupStart + n_block);
				zmmFlux = _mm512_maskz_loadu_pd(kNTail, fluxRe[equationID]+colorGroupStart+n_block);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexLeft, resAVX[equationID], 8);
				zmmSub = _mm512_sub_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], kNTail, indexLeft, zmmSub, 8);
				zmmRes = _mm512_mask_i32gather_pd(zmmZero, kNTail, indexRight, resAVX[equationID], 8);
				zmmAdd = _mm512_add_pd(zmmRes, zmmFlux);
				_mm512_mask_i32scatter_pd(resAVX[equationID], kNTail, indexRight, zmmAdd, 8);
			}//end n_tail
		}// end color loop
	}//end equationID loop
	TIMERCPU1("HostFaceLoopLoadAVXOutsideReOpt");
	//validate
	if (loopID == 0){
		for (equationID = 0; equationID < nEquation; equationID++){
			for (faceID = 0; faceID < nBoundFace; faceID++){
				le = leftCellofFace[faceID];
				if (abs(res[equationID][le]-resAVX[equationID][le])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[equationID][le], resAVX[equationID][le]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[le][offset]);
					}
					exit(0);
				}
			}
			for (faceID = nBoundFace; faceID < nTotalFace; faceID++){
				le = leftCellofFace[faceID];
				re = rightCellofFace[faceID];
				if (abs(res[equationID][le]-resAVX[equationID][le])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, le, res[equationID][le], resAVX[equationID][le]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[le][offset]);
					}
					exit(0);
				}
				if (abs(res[equationID][re]-resAVX[equationID][re])>1.0e-12) {
					printf("Error: on face %d, cell %d, res %.30e is not equal to resAVX %.30e\n", faceID, re, res[equationID][re], resAVX[equationID][re]);
					int numFaceInCell = faceNumberOfEachCell[le];
					for (int offset = 0; offset < numFaceInCell; offset++){
						printf("%d\n", cell2Face[re][offset]);
					}
					exit(0);
				}
			}
    		}
		printf("validate AVX512 faceLoadFlux outside opt  successfully\n");
	}
}
