#include "stdio.h"
#include "stdlib.h"
#include "AVX512LocalMinMax.h"
#include "GlobalVariablesHost.h"
#include "immintrin.h"
#define N_UNROLL 8
#include "newTimer.h"
using namespace newTimer;
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

        for ( int iCell = 0; iCell < nTotalCell; ++ iCell){
                //le = leftCellofFace[iFace];
		if (abs(dMinAVX[iCell] - dMin[iCell]) > 1.0e-12) {
			printf("Error: on cell %d, dMinAVX %.30e is not equal to dMin %.30e\n", iCell, dMinAVX[iCell], dMin[iCell]);
			exit(0);
		}
	}
	printf("validate AVX512 successfully\n");
}

void AVX512FaceLoopLocalMinMaxReOpt(int loopID){


}
void setAVXDMaxDMinByQNS(const int loopID){
	if (loopID == 0) printf("setAVXDMaxDMinByQNS\n");
	int iCell;
	for (iCell = 0; iCell < nTotalCell; iCell++) {
		dMinAVX[iCell] = qNS[0][iCell];
		dMaxAVX[iCell] = qNS[0][iCell];
	}
}
