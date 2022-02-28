#include "stdio.h"
#include "stdlib.h"
#include "AVX512LocalMinMax.h"
#include "GlobalVariablesHost.h"
#include "immintrin.h"
#define N_UNROLL 8
#include "newTimer.h"
using namespace newTimer;
void AVX512CellLoopLocalMinMax(int loopID){
	if (loopID == 0) printf("AVX512CellLoopLocalMinMax\n");
	int iCell;
	int * cellIDArray = (int *)malloc(nTotalCell * sizeof(int));
	cellIDArray[0] = 0;
	for (int cellID = 1; cellID < nTotalCell; cellID++){
		cellIDArray[cellID] = cellID;
	}
	int maxFaceNum = 6;
	setAVXDMaxDMinByQNS(loopID);
	__m256i zmmCellID;
	__m256i zmmNumFace;
	__m256i zmmPosiCell;
	__m256i zmmCellCellID;
	__m256i zmmNum;
	__m256i zmmPosiReal;
	__m256i zmmOffset;
	__m256i zmmOffReal;
	__m256i zmmMinusOne = _mm256_maskz_set1_epi32(0xFF, -1);
	__m256i zmmZero = _mm256_maskz_set1_epi32(0xFF, 0);
	__m512d zmmDMin;
	__m512d zmmDMax;
	__m512d zmmQNS;
	__m512d zmmTmp;
	__mmask8 zmmComp;
	__mmask8 kZero;
	__mmask8 maskTail;
	int n_block = (nTotalCell/N_UNROLL)*N_UNROLL;
	int n_tail = nTotalCell - n_block;
	TIMERCPU0("AVX512CellLoopLocalMinMax", "Cell Loop AVX512 outside");
	for (iCell = 0; iCell < n_block; iCell += N_UNROLL){
		zmmCellID = _mm256_maskz_loadu_epi32(0xFF, cellIDArray + iCell);
		zmmNumFace = _mm256_maskz_loadu_epi32(0xFF, faceNumberOfEachCell + iCell);
		zmmPosiCell = _mm256_maskz_loadu_epi32(0xFF, cell2FacePosition + iCell);
		zmmDMin = _mm512_loadu_pd(dMinAVX+iCell);
		zmmDMax = _mm512_loadu_pd(dMaxAVX+iCell);
	    	for (int offsetFace =1; offsetFace <= maxFaceNum; offsetFace++){	
			zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetFace);
			zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNumFace, zmmNum);
			kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	                zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmZero, zmmOffset);
			zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmPosiCell, zmmOffReal);

			zmmCellCellID = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmPosiReal, cell2Cell[0], 4);
			zmmQNS = _mm512_i32gather_pd(zmmCellCellID, qNS[0], 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMin, zmmQNS, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNS, zmmDMin);
			zmmComp = _mm512_cmp_pd_mask(zmmDMax, zmmQNS, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNS, zmmDMax);
            	}
        		_mm512_storeu_pd(dMinAVX+iCell, zmmDMin);
	        	_mm512_storeu_pd(dMaxAVX+iCell, zmmDMax);
	}
	if (n_tail > 0){
		maskTail = 0xFF>>(8-n_tail);
		__m512d zmmTmp = _mm512_setzero_pd();
		zmmCellID = _mm256_maskz_loadu_epi32(maskTail, cellIDArray + n_block);
		zmmNumFace = _mm256_maskz_loadu_epi32(maskTail, faceNumberOfEachCell + n_block);
		zmmPosiCell = _mm256_maskz_loadu_epi32(maskTail, cell2FacePosition + n_block);
	    	zmmDMin = _mm512_maskz_loadu_pd(maskTail ,dMinAVX + n_block);
	    	zmmDMax = _mm512_maskz_loadu_pd(maskTail ,dMaxAVX + n_block);
	    	for (int offsetFace =1; offsetFace <= maxFaceNum; offsetFace++){	
			zmmNum = _mm256_maskz_set1_epi32(maskTail, offsetFace);
			zmmOffset = _mm256_maskz_sub_epi32(maskTail, zmmNumFace, zmmNum);
			kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	                zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmZero, zmmOffset);
			zmmPosiReal = _mm256_maskz_add_epi32(maskTail, zmmPosiCell, zmmOffReal);

			zmmCellCellID = _mm256_mmask_i32gather_epi32(zmmZero, maskTail, zmmPosiReal, cell2Cell[0], 4);
			zmmQNS = _mm512_mask_i32gather_pd(zmmTmp, maskTail, zmmCellCellID, qNS[0], 8);
			zmmComp = _mm512_mask_cmp_pd_mask(maskTail, zmmDMin, zmmQNS, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNS, zmmDMin);
			zmmComp = _mm512_mask_cmp_pd_mask(maskTail, zmmDMax, zmmQNS, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNS, zmmDMax);
		}
	    		_mm512_mask_compressstoreu_pd(dMinAVX + n_block, maskTail, zmmDMin);
	    		_mm512_mask_compressstoreu_pd(dMaxAVX + n_block, maskTail, zmmDMax);


	}
	TIMERCPU1("AVX512CellLoopLocalMinMax");
	if (loopID == 0) {
	    for ( int iCell = 0; iCell < nTotalCell; ++ iCell){
		if (abs(dMinAVX[iCell] - dMin[iCell]) > 1.0e-12) {
			printf("Error: nTotalCell %d, on cell %d, dMinAVX %.30e is not equal to dMin %.30e\n", nTotalCell, iCell, dMinAVX[iCell], dMin[iCell]);
			exit(0);
		}
		if (abs(dMaxAVX[iCell] - dMax[iCell]) > 1.0e-12) {
			printf("Error: nTotalCell %d, on cell %d, dMaxAVX %.30e is not equal to dMax %.30e\n", nTotalCell, iCell, dMaxAVX[iCell], dMax[iCell]);
			exit(0);
		}
	    }	
	    printf("validate AVX512 Cell Loop ReOpt successfully\n");
	}

}
void AVX512FaceLoopLocalMinMax(int loopID){
	if (loopID == 0) printf("AVX512FaceLoopLocalMinMax\n");
	int iFace;
	int le, re;
	setAVXDMaxDMinByQNS(loopID);
	__m256i zmmLeftCell;
	__m256i zmmFaceID;
	__m256i zmmRightCell;
	__m512d zmmDMin;
	__m512d zmmDMax;
	__m512d zmmQNSRight;
	__m512d zmmQNSLeft;
	__mmask8 zmmComp;
	__m256i zmmZero = _mm256_maskz_set1_epi32(0xFF, 0);
	//AVX512 boundary faces
	TIMERCPU0("AVX512FaceLoopMinMax", "AVX512Face Loop MinMax");
	for (int colorID = 0; colorID < BoundFaceColorNum; colorID++){
		int colorGroupNum = BoundFaceGroupNum[colorID];
                int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
                int n_tail = colorGroupNum - n_block;
                int colorGroupStart = BoundFaceGroupPosi[colorID];
		for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
			zmmFaceID = _mm256_maskz_loadu_epi32(0xFF, BoundFaceGroup + colorGroupStart + colorGroupID);
			zmmLeftCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, leftCellofFace, 4);
			zmmRightCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, rightCellofFace, 4);
			zmmDMin = _mm512_i32gather_pd(zmmLeftCell, dMinAVX, 8);
			zmmDMax = _mm512_i32gather_pd(zmmLeftCell, dMaxAVX, 8);
			zmmQNSRight = _mm512_i32gather_pd(zmmRightCell, qNS[0], 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMin, zmmQNSRight, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMin);
			_mm512_i32scatter_pd(dMinAVX, zmmLeftCell, zmmDMin, 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMax, zmmQNSRight, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMax);
			_mm512_i32scatter_pd(dMaxAVX, zmmLeftCell, zmmDMax, 8);
		}

		if (n_tail > 0){
			__m512d zmmTmp = _mm512_setzero_pd();
			zmmFaceID = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), BoundFaceGroup + colorGroupStart + n_block);
			zmmLeftCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF>>(8-n_tail), zmmFaceID, leftCellofFace, 4);
			zmmRightCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF>>(8-n_tail), zmmFaceID, rightCellofFace, 4);
			zmmDMin = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, dMinAVX, 8);
			zmmDMax = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, dMaxAVX, 8);
			zmmQNSRight = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmRightCell, qNS[0], 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMin, zmmQNSRight, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMin);
			_mm512_mask_i32scatter_pd(dMinAVX, 0xFF>>(8-n_tail), zmmLeftCell, zmmDMin, 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMax, zmmQNSRight, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMax);
			_mm512_mask_i32scatter_pd(dMaxAVX, 0xFF>>(8-n_tail), zmmLeftCell, zmmDMax, 8);
			
		}
	}

	//for AVX512 Interior faces
	for (int colorID = 0; colorID < InteriorFaceColorNum; colorID++){
		int colorGroupNum = InteriorFaceGroupNum[colorID];
                int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
                int n_tail = colorGroupNum - n_block;
                int colorGroupStart = InteriorFaceGroupPosi[colorID];
		for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
			zmmFaceID = _mm256_maskz_loadu_epi32(0xFF, InteriorFaceGroup + colorGroupStart + colorGroupID);
			zmmLeftCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, leftCellofFace, 4);
			zmmRightCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, rightCellofFace, 4);
			zmmDMin = _mm512_i32gather_pd(zmmLeftCell, dMinAVX, 8);
			zmmDMax = _mm512_i32gather_pd(zmmLeftCell, dMaxAVX, 8);
			zmmQNSRight = _mm512_i32gather_pd(zmmRightCell, qNS[0], 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMin, zmmQNSRight, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMin);
			_mm512_i32scatter_pd(dMinAVX, zmmLeftCell, zmmDMin, 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMax, zmmQNSRight, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMax);
			_mm512_i32scatter_pd(dMaxAVX, zmmLeftCell, zmmDMax, 8);

			zmmDMin = _mm512_i32gather_pd(zmmRightCell, dMinAVX, 8);
			zmmDMax = _mm512_i32gather_pd(zmmRightCell, dMaxAVX, 8);
			zmmQNSLeft = _mm512_i32gather_pd(zmmLeftCell, qNS[0], 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMin, zmmQNSLeft, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSLeft, zmmDMin);
			_mm512_i32scatter_pd(dMinAVX, zmmRightCell, zmmDMin, 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMax, zmmQNSLeft, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSLeft, zmmDMax);
			_mm512_i32scatter_pd(dMaxAVX, zmmRightCell, zmmDMax, 8);
		}

		if (n_tail > 0){
			__m512d zmmTmp = _mm512_setzero_pd();
			zmmFaceID = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), InteriorFaceGroup + colorGroupStart + n_block);
			zmmLeftCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF>>(8-n_tail), zmmFaceID, leftCellofFace, 4);
			zmmRightCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF>>(8-n_tail), zmmFaceID, rightCellofFace, 4);
			zmmDMin = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, dMinAVX, 8);
			zmmDMax = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, dMaxAVX, 8);
			zmmQNSRight = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmRightCell, qNS[0], 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMin, zmmQNSRight, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMin);
			_mm512_mask_i32scatter_pd(dMinAVX, 0xFF>>(8-n_tail), zmmLeftCell, zmmDMin, 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMax, zmmQNSRight, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMax);
			_mm512_mask_i32scatter_pd(dMaxAVX, 0xFF>>(8-n_tail), zmmLeftCell, zmmDMax, 8);
			
			zmmDMin = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmRightCell, dMinAVX, 8);
			zmmDMax = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmRightCell, dMaxAVX, 8);
			zmmQNSLeft = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, qNS[0], 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMin, zmmQNSLeft, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSLeft, zmmDMin);
			_mm512_mask_i32scatter_pd(dMinAVX, 0xFF>>(8-n_tail), zmmRightCell, zmmDMin, 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMax, zmmQNSLeft, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSLeft, zmmDMax);
			_mm512_mask_i32scatter_pd(dMaxAVX, 0xFF>>(8-n_tail), zmmRightCell, zmmDMax, 8);
		}
	}
	TIMERCPU1("AVX512FaceLoopMinMax");

	if (loopID == 0) {
        	for ( int iCell = 0; iCell < nTotalCell; ++ iCell){
			if (abs(dMinAVX[iCell] - dMin[iCell]) > 1.0e-12) {
				printf("Error: on cell %d, dMinAVX %.30e is not equal to dMin %.30e\n", iCell, dMinAVX[iCell], dMin[iCell]);
				exit(0);
			}
			if (abs(dMaxAVX[iCell] - dMax[iCell]) > 1.0e-12) {
				printf("Error: on cell %d, dMaxAVX %.30e is not equal to dMax %.30e\n", iCell, dMaxAVX[iCell], dMax[iCell]);
				exit(0);
			}
		}
		printf("validate AVX512 successfully\n");
	}
}

void AVX512FaceLoopLocalMinMaxReOpt(int loopID){
	if (loopID == 0) printf("AVX512FaceLoopLocalMinMaxReOpt\n");
	int iFace;
	int le, re;
	setAVXDMaxDMinByQNS(loopID);
	__m256i zmmLeftCell;
	//__m256i zmmFaceID;
	__m256i zmmRightCell;
	__m512d zmmDMin;
	__m512d zmmDMax;
	__m512d zmmQNSRight;
	__m512d zmmQNSLeft;
	__mmask8 zmmComp;
	__m256i zmmZero = _mm256_maskz_set1_epi32(0xFF, 0);
	//AVX512 boundary faces
	TIMERCPU0("AVX512FaceLoopMinMaxReOpt", "ReOptAVX512Face Loop MinMax");
	for (int colorID = 0; colorID < BoundFaceColorNum; colorID++){
		int colorGroupNum = BoundFaceGroupNum[colorID];
                int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
                int n_tail = colorGroupNum - n_block;
                int colorGroupStart = BoundFaceGroupPosi[colorID];
		for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
			//zmmFaceID = _mm256_maskz_loadu_epi32(0xFF, BoundFaceGroup + colorGroupStart + colorGroupID);
			zmmLeftCell = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceRe + colorGroupStart + colorGroupID);
			//indexLeft = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceRe + colorGroupStart + colorGroupID);
			zmmRightCell = _mm256_maskz_loadu_epi32(0xFF, rightCellofFaceRe + colorGroupStart + colorGroupID);
			zmmDMin = _mm512_i32gather_pd(zmmLeftCell, dMinAVX, 8);
			zmmDMax = _mm512_i32gather_pd(zmmLeftCell, dMaxAVX, 8);
			zmmQNSRight = _mm512_i32gather_pd(zmmRightCell, qNS[0], 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMin, zmmQNSRight, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMin);
			_mm512_i32scatter_pd(dMinAVX, zmmLeftCell, zmmDMin, 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMax, zmmQNSRight, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMax);
			_mm512_i32scatter_pd(dMaxAVX, zmmLeftCell, zmmDMax, 8);
		}

		if (n_tail > 0){
			__m512d zmmTmp = _mm512_setzero_pd();
			//zmmFaceID = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), BoundFaceGroup + colorGroupStart + n_block);
			//zmmLeftCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF>>(8-n_tail), zmmFaceID, leftCellofFace, 4);
			//zmmRightCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF>>(8-n_tail), zmmFaceID, rightCellofFace, 4);
			zmmLeftCell = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), leftCellofFaceRe + colorGroupStart + n_block);
			zmmRightCell = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), rightCellofFaceRe + colorGroupStart + n_block);
			zmmDMin = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, dMinAVX, 8);
			zmmDMax = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, dMaxAVX, 8);
			zmmQNSRight = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmRightCell, qNS[0], 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMin, zmmQNSRight, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMin);
			_mm512_mask_i32scatter_pd(dMinAVX, 0xFF>>(8-n_tail), zmmLeftCell, zmmDMin, 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMax, zmmQNSRight, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMax);
			_mm512_mask_i32scatter_pd(dMaxAVX, 0xFF>>(8-n_tail), zmmLeftCell, zmmDMax, 8);
			
		}
	}

	//for AVX512 Interior faces
	for (int colorID = 0; colorID < InteriorFaceColorNum; colorID++){
		int colorGroupNum = InteriorFaceGroupNum[colorID];
                int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
                int n_tail = colorGroupNum - n_block;
                int colorGroupStart = InteriorFaceGroupPosi[colorID]+nBoundFace;
		for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
			//zmmFaceID = _mm256_maskz_loadu_epi32(0xFF, InteriorFaceGroup + colorGroupStart + colorGroupID);
			//zmmLeftCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, leftCellofFace, 4);
			//zmmRightCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF, zmmFaceID, rightCellofFace, 4);
			zmmLeftCell = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceRe + colorGroupStart + colorGroupID);
			zmmRightCell = _mm256_maskz_loadu_epi32(0xFF, rightCellofFaceRe + colorGroupStart + colorGroupID);
			zmmDMin = _mm512_i32gather_pd(zmmLeftCell, dMinAVX, 8);
			zmmDMax = _mm512_i32gather_pd(zmmLeftCell, dMaxAVX, 8);
			zmmQNSRight = _mm512_i32gather_pd(zmmRightCell, qNS[0], 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMin, zmmQNSRight, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMin);
			_mm512_i32scatter_pd(dMinAVX, zmmLeftCell, zmmDMin, 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMax, zmmQNSRight, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMax);
			_mm512_i32scatter_pd(dMaxAVX, zmmLeftCell, zmmDMax, 8);

			zmmDMin = _mm512_i32gather_pd(zmmRightCell, dMinAVX, 8);
			zmmDMax = _mm512_i32gather_pd(zmmRightCell, dMaxAVX, 8);
			zmmQNSLeft = _mm512_i32gather_pd(zmmLeftCell, qNS[0], 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMin, zmmQNSLeft, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSLeft, zmmDMin);
			_mm512_i32scatter_pd(dMinAVX, zmmRightCell, zmmDMin, 8);
			zmmComp = _mm512_cmp_pd_mask(zmmDMax, zmmQNSLeft, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSLeft, zmmDMax);
			_mm512_i32scatter_pd(dMaxAVX, zmmRightCell, zmmDMax, 8);
		}

		if (n_tail > 0){
			__m512d zmmTmp = _mm512_setzero_pd();
			//zmmFaceID = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), InteriorFaceGroup + colorGroupStart + n_block);
			//zmmLeftCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF>>(8-n_tail), zmmFaceID, leftCellofFace, 4);
			//zmmRightCell = _mm256_mmask_i32gather_epi32(zmmZero, 0xFF>>(8-n_tail), zmmFaceID, rightCellofFace, 4);
			zmmLeftCell = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), leftCellofFaceRe + colorGroupStart + n_block);
			zmmRightCell = _mm256_maskz_loadu_epi32(0xFF>>(8-n_tail), rightCellofFaceRe + colorGroupStart + n_block);
			zmmDMin = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, dMinAVX, 8);
			zmmDMax = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, dMaxAVX, 8);
			zmmQNSRight = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmRightCell, qNS[0], 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMin, zmmQNSRight, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMin);
			_mm512_mask_i32scatter_pd(dMinAVX, 0xFF>>(8-n_tail), zmmLeftCell, zmmDMin, 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMax, zmmQNSRight, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSRight, zmmDMax);
			_mm512_mask_i32scatter_pd(dMaxAVX, 0xFF>>(8-n_tail), zmmLeftCell, zmmDMax, 8);
			
			zmmDMin = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmRightCell, dMinAVX, 8);
			zmmDMax = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmRightCell, dMaxAVX, 8);
			zmmQNSLeft = _mm512_mask_i32gather_pd(zmmTmp, 0xFF>>(8-n_tail), zmmLeftCell, qNS[0], 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMin, zmmQNSLeft, _CMP_LT_OQ);
			zmmDMin = _mm512_mask_blend_pd(zmmComp, zmmQNSLeft, zmmDMin);
			_mm512_mask_i32scatter_pd(dMinAVX, 0xFF>>(8-n_tail), zmmRightCell, zmmDMin, 8);
			zmmComp = _mm512_mask_cmp_pd_mask(0xFF>>(8-n_tail), zmmDMax, zmmQNSLeft, _CMP_GT_OQ);
			zmmDMax = _mm512_mask_blend_pd(zmmComp, zmmQNSLeft, zmmDMax);
			_mm512_mask_i32scatter_pd(dMaxAVX, 0xFF>>(8-n_tail), zmmRightCell, zmmDMax, 8);
		}
	}
	TIMERCPU1("AVX512FaceLoopMinMaxReOpt");

	if (loopID == 0) {
        	for ( int iCell = 0; iCell < nTotalCell; ++ iCell){
			if (abs(dMinAVX[iCell] - dMin[iCell]) > 1.0e-12) {
				printf("Error: on cell %d, dMinAVX %.30e is not equal to dMin %.30e\n", iCell, dMinAVX[iCell], dMin[iCell]);
				exit(0);
			}
			if (abs(dMaxAVX[iCell] - dMax[iCell]) > 1.0e-12) {
				printf("Error: on cell %d, dMaxAVX %.30e is not equal to dMax %.30e\n", iCell, dMaxAVX[iCell], dMax[iCell]);
				exit(0);
			}
		}
		printf("validate AVX512 ReOpt successfully\n");
	}
}
void setAVXDMaxDMinByQNS(const int loopID){
	if (loopID == 0) printf("setAVXDMaxDMinByQNS\n");
	int iCell;
	for (iCell = 0; iCell < nTotalCell; iCell++) {
		dMinAVX[iCell] = qNS[0][iCell];
		dMaxAVX[iCell] = qNS[0][iCell];
	}
}
