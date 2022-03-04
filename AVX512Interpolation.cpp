#include "stdio.h"
#include "AVX512Interpolation.h"
#include "GlobalVariablesHost.h"
#include "constantVars.hxx"
#include "newTimer.h"
#include "stdlib.h"
#include "immintrin.h"
#define N_UNROLL 8
using namespace newTimer;
void initNCountQNodeTNodeAVXZero(const int loopID){
	if (loopID == 0) printf("AVX512initNCountQNodeTNodeZero\n");
	if ((!tNode)||(!qNode)||(!nCount)) {
		printf("Error: tCell, qNode, and nCount should be allodated memory\n");
		exit(0);
	}
	int nEquation = nl + nchem;
	int iNode;
	int iEquation;
	for (iNode = 0; iNode < nTotalNode; iNode++){
		for (iEquation = 0; iEquation < nEquation; iEquation++){
			qNodeAVX[iEquation][iNode] = 0.0;
		}
		for (iEquation = 0; iEquation < nTemperature; iEquation++){
			tNodeAVX[iEquation][iNode] = 0.0;
		}
		nCountAVX[iNode] = 0;
	}
}
void AVX512FaceLoopSepInterpolation(int loopID){
	if (loopID == 0) printf("AVX512FaceLoopInterpolation\n");

	initNCountQNodeTNodeAVXZero(loopID);
	int nEquation = nl + nchem;

	//For AVX512
	int equationID;
	int maxNodeNum = 4;//maximum number of node in a face
	__m256i zmmLeftCell;
	__m256i zmmRightCell;
	__m256i zmmFace2NodePosition;
	__m256i zmmNodeNumberOfEachFace;
	__m256i zmmNum;
	__m256i zmmOffset;
	__m256i zmmOffReal;
	__m256i zmmPosiReal;
	__m256i zmmNodeID;
	__m256i zmmNCount;
	__m256i zmmNCPlusOne;
	__m256i zmmIndexZero = _mm256_maskz_set1_epi32(0xFF, 0);
	__m256i zmmMinusOne = _mm256_maskz_set1_epi32(0xFF, -1);
	__m256i zmmPositiveOne = _mm256_maskz_set1_epi32(0xFF, 1);
	__m256i zmmPositiveTwo = _mm256_maskz_set1_epi32(0xFF, 2);
	__m512d zmmZero = _mm512_setzero_pd();

	__m512d zmmQNS;
	__m512d zmmQNSRight;
	__m512d zmmQNode;
	__m512d zmmAdd;

	__mmask8 kZero;
	TIMERCPU0("AVX512FaceLoopSepNCQTNodeSep", "AVX512 Separate face loop for NCQTNode");
	//Interpolation of qNode
        for (equationID = 0; equationID < nEquation; equationID++){
                for (int colorID = 0; colorID < BoundFaceNodeColorNum; colorID++){
                        int colorGroupNum = BoundFaceNodeGroupNum[colorID];
                        int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
                        int n_tail = colorGroupNum - n_block;
                        int colorGroupStart = BoundFaceNodeGroupPosi[colorID];
                        for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){	
				zmmLeftCell = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceNodeRe + colorGroupStart + colorGroupID);
				zmmQNS = _mm512_i32gather_pd(zmmLeftCell, qNS[equationID], 8);
				zmmFace2NodePosition = _mm256_maskz_loadu_epi32(0xFF, face2NodePositionNodeRe + colorGroupStart + colorGroupID);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(0xFF, nodeNumberOfEachFaceNodeRe + colorGroupStart + colorGroupID);
				for (int offsetNode = 1; offsetNode <= maxNodeNum; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	        		        zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmFace2NodePosition, zmmOffReal);
					//zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2NodeNodeRe, 4);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2Node, 4);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, qNodeAVX[equationID], 8);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNS, zmmQNode);
					_mm512_i32scatter_pd(qNodeAVX[equationID], zmmNodeID, zmmAdd, 8);
				}

			}
			if (n_tail > 0){
				__mmask8 kNTail = 0xFF>>(8-n_tail);
				zmmLeftCell = _mm256_maskz_loadu_epi32(kNTail, leftCellofFaceNodeRe + colorGroupStart + n_block);
				zmmQNS = _mm512_mask_i32gather_pd(zmmZero, kNTail, zmmLeftCell, qNS[equationID], 8);
				zmmFace2NodePosition = _mm256_maskz_loadu_epi32(kNTail, face2NodePositionNodeRe + colorGroupStart + n_block);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(kNTail, nodeNumberOfEachFaceNodeRe + colorGroupStart + n_block);
				for (int offsetNode = 1; offsetNode <= maxNodeNum; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	        		        zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmFace2NodePosition, zmmOffReal);
					//zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2NodeNodeRe, 4);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmPosiReal, face2Node, 4);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, qNodeAVX[equationID], 8);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNS, zmmQNode);
					 _mm512_mask_i32scatter_pd(qNodeAVX[equationID], kNTail, zmmNodeID, zmmAdd, 8);
				}
				

			}
		}
	}
        for (equationID = 0; equationID < nEquation; equationID++){
                for (int colorID = 0; colorID < InteriorFaceNodeColorNum; colorID++){
                        int colorGroupNum = InteriorFaceNodeGroupNum[colorID];
                        int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
                        int n_tail = colorGroupNum - n_block;
                        int colorGroupStart = InteriorFaceNodeGroupPosi[colorID] + nBoundFace;
                        for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){	
				zmmLeftCell = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceNodeRe + colorGroupStart + colorGroupID);
				zmmRightCell = _mm256_maskz_loadu_epi32(0xFF, rightCellofFaceNodeRe + colorGroupStart + colorGroupID);
				zmmQNS = _mm512_i32gather_pd(zmmLeftCell, qNS[equationID], 8);
				zmmQNSRight = _mm512_i32gather_pd(zmmRightCell, qNS[equationID], 8);
				zmmFace2NodePosition = _mm256_maskz_loadu_epi32(0xFF, face2NodePositionNodeRe + colorGroupStart + colorGroupID);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(0xFF, nodeNumberOfEachFaceNodeRe + colorGroupStart + colorGroupID);
				for (int offsetNode = 1; offsetNode <= maxNodeNum; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	        		        zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmFace2NodePosition, zmmOffReal);
					//zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2NodeNodeRe, 4);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2Node, 4);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, qNodeAVX[equationID], 8);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNS, zmmQNode);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNSRight, zmmAdd);
					_mm512_i32scatter_pd(qNodeAVX[equationID], zmmNodeID, zmmAdd, 8);
				}

			}
			if (n_tail > 0){
				__mmask8 kNTail = 0xFF>>(8-n_tail);
				zmmLeftCell = _mm256_maskz_loadu_epi32(kNTail, leftCellofFaceNodeRe + colorGroupStart + n_block);
				zmmRightCell = _mm256_maskz_loadu_epi32(kNTail, rightCellofFaceNodeRe + colorGroupStart + n_block);
				zmmQNS = _mm512_mask_i32gather_pd(zmmZero, kNTail, zmmLeftCell, qNS[equationID], 8);
				zmmQNSRight = _mm512_mask_i32gather_pd(zmmZero, kNTail, zmmRightCell, qNS[equationID], 8);
				zmmFace2NodePosition = _mm256_maskz_loadu_epi32(kNTail, face2NodePositionNodeRe + colorGroupStart + n_block);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(kNTail, nodeNumberOfEachFaceNodeRe + colorGroupStart + n_block);
				for (int offsetNode = 1; offsetNode <= maxNodeNum; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	        		        zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmFace2NodePosition, zmmOffReal);
					//zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2NodeNodeRe, 4);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmPosiReal, face2Node, 4);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, qNodeAVX[equationID], 8);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNS, zmmQNode);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNSRight, zmmAdd);
					 _mm512_mask_i32scatter_pd(qNodeAVX[equationID], kNTail, zmmNodeID, zmmAdd, 8);
				}
				

			}
		}
	}

		//Interpolation of tNode and nCount
                for (int colorID = 0; colorID < BoundFaceNodeColorNum; colorID++){
                        int colorGroupNum = BoundFaceNodeGroupNum[colorID];
                        int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
                        int n_tail = colorGroupNum - n_block;
                        int colorGroupStart = BoundFaceNodeGroupPosi[colorID];
                        for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){	
				zmmLeftCell = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceNodeRe + colorGroupStart + colorGroupID);
				zmmQNS = _mm512_i32gather_pd(zmmLeftCell, tCell[0], 8);
				zmmFace2NodePosition = _mm256_maskz_loadu_epi32(0xFF, face2NodePositionNodeRe + colorGroupStart + colorGroupID);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(0xFF, nodeNumberOfEachFaceNodeRe + colorGroupStart + colorGroupID);
				for (int offsetNode = 1; offsetNode <= maxNodeNum; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	        		        zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmFace2NodePosition, zmmOffReal);
					//zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2NodeNodeRe, 4);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2Node, 4);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, tNodeAVX[0], 8);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNS, zmmQNode);
					_mm512_i32scatter_pd(tNodeAVX[0], zmmNodeID, zmmAdd, 8);
					zmmNCount = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmNodeID, nCountAVX, 4);
					zmmNCPlusOne = _mm256_mask_add_epi32(zmmNCount, kZero, zmmPositiveOne, zmmNCount);
					_mm256_i32scatter_epi32(nCountAVX, zmmNodeID, zmmNCPlusOne, 4);
				}

			}
			if (n_tail > 0){
				__mmask8 kNTail = 0xFF>>(8-n_tail);
				zmmLeftCell = _mm256_maskz_loadu_epi32(kNTail, leftCellofFaceNodeRe + colorGroupStart + n_block);
				zmmQNS = _mm512_mask_i32gather_pd(zmmZero, kNTail, zmmLeftCell, tCell[0], 8);
				zmmFace2NodePosition = _mm256_maskz_loadu_epi32(kNTail, face2NodePositionNodeRe + colorGroupStart + n_block);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(kNTail, nodeNumberOfEachFaceNodeRe + colorGroupStart + n_block);
				for (int offsetNode = 1; offsetNode <= maxNodeNum; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	        		        zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmFace2NodePosition, zmmOffReal);
					//zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2NodeNodeRe, 4);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmPosiReal, face2Node, 4);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, tNodeAVX[0], 8);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNS, zmmQNode);
					 _mm512_mask_i32scatter_pd(tNodeAVX[0], kNTail, zmmNodeID, zmmAdd, 8);
					zmmNCount = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmNodeID, nCountAVX, 4);
					zmmNCPlusOne = _mm256_mask_add_epi32(zmmNCount, kZero, zmmPositiveOne, zmmNCount);
					_mm256_mask_i32scatter_epi32(nCountAVX, kNTail, zmmNodeID, zmmNCPlusOne, 4);
				}
				

			}
		}
                for (int colorID = 0; colorID < InteriorFaceNodeColorNum; colorID++){
                        int colorGroupNum = InteriorFaceNodeGroupNum[colorID];
                        int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
                        int n_tail = colorGroupNum - n_block;
                        int colorGroupStart = InteriorFaceNodeGroupPosi[colorID] + nBoundFace;
                        for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){	
				zmmLeftCell = _mm256_maskz_loadu_epi32(0xFF, leftCellofFaceNodeRe + colorGroupStart + colorGroupID);
				zmmRightCell = _mm256_maskz_loadu_epi32(0xFF, rightCellofFaceNodeRe + colorGroupStart + colorGroupID);
				zmmQNS = _mm512_i32gather_pd(zmmLeftCell, tCell[0], 8);
				zmmQNSRight = _mm512_i32gather_pd(zmmRightCell, tCell[0], 8);
				zmmFace2NodePosition = _mm256_maskz_loadu_epi32(0xFF, face2NodePositionNodeRe + colorGroupStart + colorGroupID);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(0xFF, nodeNumberOfEachFaceNodeRe + colorGroupStart + colorGroupID);
				for (int offsetNode = 1; offsetNode <= maxNodeNum; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	        		        zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmFace2NodePosition, zmmOffReal);
					//zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2NodeNodeRe, 4);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2Node, 4);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, tNodeAVX[0], 8);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNS, zmmQNode);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNSRight, zmmAdd);
					_mm512_i32scatter_pd(tNodeAVX[0], zmmNodeID, zmmAdd, 8);
					zmmNCount = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmNodeID, nCountAVX, 4);
					zmmNCPlusOne = _mm256_mask_add_epi32(zmmNCount, kZero, zmmPositiveTwo, zmmNCount);
					_mm256_i32scatter_epi32(nCountAVX, zmmNodeID, zmmNCPlusOne, 4);
				}

			}
			if (n_tail > 0){
				__mmask8 kNTail = 0xFF>>(8-n_tail);
				zmmLeftCell = _mm256_maskz_loadu_epi32(kNTail, leftCellofFaceNodeRe + colorGroupStart + n_block);
				zmmRightCell = _mm256_maskz_loadu_epi32(kNTail, rightCellofFaceNodeRe + colorGroupStart + n_block);
				zmmQNS = _mm512_mask_i32gather_pd(zmmZero, kNTail, zmmLeftCell, tCell[0], 8);
				zmmQNSRight = _mm512_mask_i32gather_pd(zmmZero, kNTail, zmmRightCell, tCell[0], 8);
				zmmFace2NodePosition = _mm256_maskz_loadu_epi32(kNTail, face2NodePositionNodeRe + colorGroupStart + n_block);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(kNTail, nodeNumberOfEachFaceNodeRe + colorGroupStart + n_block);
				for (int offsetNode = 1; offsetNode <= maxNodeNum; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
	        		        zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmFace2NodePosition, zmmOffReal);
					//zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, face2NodeNodeRe, 4);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmPosiReal, face2Node, 4);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, tNodeAVX[0], 8);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNS, zmmQNode);
					zmmAdd = _mm512_mask_add_pd(zmmQNode, kZero, zmmQNSRight, zmmAdd);
					 _mm512_mask_i32scatter_pd(tNodeAVX[0], kNTail, zmmNodeID, zmmAdd, 8);
					zmmNCount = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmNodeID, nCountAVX, 4);
					zmmNCPlusOne = _mm256_mask_add_epi32(zmmNCount, kZero, zmmPositiveTwo, zmmNCount);
					_mm256_mask_i32scatter_epi32(nCountAVX, kNTail, zmmNodeID, zmmNCPlusOne, 4);
				}
				

			}
		}
	TIMERCPU1("AVX512FaceLoopSepNCQTNodeSep");

	//Validation
     if (loopID == 0){
	for (equationID = 0; equationID < nEquation; equationID++){
		for (int faceID = 0; faceID < nTotalFace; faceID++){
			int nodeNum = nodeNumberOfEachFace[faceID];
			int nodePosi = face2NodePosition[faceID];
			for (int offset = 0; offset < nodeNum; offset++){
				int nodeID = face2Node[nodePosi + offset];
				if (abs(qNodeAVX[equationID][nodeID] - qNode[equationID][nodeID]) >  1.0e-12) {
					printf("Error: nBoundFace = %d, faceID = %d, nodeID = %d, qNodeAVX = %.30e, qNode = %.30e\n", nBoundFace, faceID, nodeID, qNodeAVX[0][nodeID], qNode[0][nodeID]);
					exit(0);
				}
			}
		}
	}
		for (int faceID = 0; faceID < nTotalFace; faceID++){
			int nodeNum = nodeNumberOfEachFace[faceID];
			int nodePosi = face2NodePosition[faceID];
			for (int offset = 0; offset < nodeNum; offset++){
				int nodeID = face2Node[nodePosi + offset];
				if (abs(tNodeAVX[0][nodeID] - tNode[0][nodeID]) >  1.0e-12) {
					printf("Error: nBoundFace = %d, faceID = %d, nodeID = %d, tNodeAVX = %.30e, tNode = %.30e\n", nBoundFace, faceID, nodeID, tNodeAVX[0][nodeID], tNode[0][nodeID]);
					exit(0);
				}
				if (abs(nCountAVX[nodeID] - nCount[nodeID]) >  1.0e-12) {
					printf("Error: nBoundFace = %d, faceID = %d, nodeID = %d, nCountAVX = %.30e, nCount = %.30e\n", nBoundFace, faceID, nodeID, nCountAVX[nodeID], nCount[nodeID]);
					exit(0);
				}
			}
		}
		printf("Test AVX512 of qNode, tNode and nCount on whole faces successfully\n");
      }//end if validation

}

void AVX512CellLoopSepInterpolation(int loopID){
	if (loopID == 0) printf("AVX512CellLoopInterpolation\n");
	
	initNCountQNodeTNodeAVXZero(loopID);

	int nEquation = nl + nchem;
	int maxNodes = 8;
	__m256i zmmNodePosi;
	__m256i zmmNodeNumberOfEachFace;
	__m256i zmmNum;
	__m256i zmmOffset;
	__m256i zmmOffReal;
	__m256i zmmPosiReal;
	__m256i zmmNodeID;
	__m256i zmmNCount;
	__m256i zmmAccessInt;
	__m256i zmmMinusOne = _mm256_maskz_set1_epi32(0xFF, -1);
	__m256i zmmIndexZero = _mm256_maskz_set1_epi32(0xFF, 0);

	__m512d zmmQNS;
	__m512d zmmAccessFrequency;
	__m512d zmmQNode;
	__m512d zmmFmad;

	__mmask8 kZero;
	__mmask8 kNTail;
	//AVX512 for qNode
	TIMERCPU0("AVX512CellLoopSepNCQTNodeSep", "AVX512 Separate cell loop for NCQTNode");
	for (int equationID = 0; equationID < nEquation; equationID++){
		for (int colorID = 0; colorID < CellNodeColorNum; colorID++){
			int colorGroupNum = CellNodeGroupNum[colorID];
			int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
			int n_tail = colorGroupNum - n_block;
			int colorGroupStart = CellNodeGroupPosi[colorID];
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				zmmNodePosi = _mm256_maskz_loadu_epi32(0xFF, cell2NodePositionRe + colorGroupStart + colorGroupID);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(0xFF, nodeNumberOfEachCellRe + colorGroupStart + colorGroupID);
				zmmQNS = _mm512_loadu_pd(qNSRe[equationID]+ colorGroupStart + colorGroupID);
				for (int offsetNode = 1; offsetNode <= maxNodes; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
					zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmNodePosi, zmmOffReal);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, cell2Node, 4);
					zmmAccessFrequency = _mm512_i32gather_pd(zmmPosiReal, cell2NodeCountRe, 8);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, qNodeAVX[equationID], 8);
					zmmFmad = _mm512_mask3_fmadd_pd(zmmQNS, zmmAccessFrequency, zmmQNode, kZero);
					_mm512_i32scatter_pd(qNodeAVX[equationID], zmmNodeID, zmmFmad, 8);
				}
			}
			if (n_tail > 0){
				kNTail = 0xFF>>(8-n_tail);
				zmmNodePosi = _mm256_maskz_loadu_epi32(kNTail, cell2NodePositionRe + colorGroupStart + n_block);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(kNTail, nodeNumberOfEachCellRe + colorGroupStart + n_block);
				zmmQNS = _mm512_maskz_loadu_pd(kNTail, qNSRe[equationID]+ colorGroupStart + n_block);
				for (int offsetNode = 1; offsetNode <= maxNodes; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
					zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmNodePosi, zmmOffReal);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmPosiReal, cell2Node, 4);
					zmmAccessFrequency = _mm512_i32gather_pd(zmmPosiReal, cell2NodeCountRe, 8);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, qNodeAVX[equationID], 8);
					zmmFmad = _mm512_mask3_fmadd_pd(zmmQNS, zmmAccessFrequency, zmmQNode, kZero);
					_mm512_mask_i32scatter_pd(qNodeAVX[equationID], kNTail, zmmNodeID, zmmFmad, 8);

				}
			}
		}
	}
	//AVX512 for tNode and nCount
		for (int colorID = 0; colorID < CellNodeColorNum; colorID++){
			int colorGroupNum = CellNodeGroupNum[colorID];
			int n_block = (colorGroupNum/N_UNROLL)*N_UNROLL;
			int n_tail = colorGroupNum - n_block;
			int colorGroupStart = CellNodeGroupPosi[colorID];
			for (int colorGroupID = 0; colorGroupID < n_block; colorGroupID+=N_UNROLL){
				zmmNodePosi = _mm256_maskz_loadu_epi32(0xFF, cell2NodePositionRe + colorGroupStart + colorGroupID);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(0xFF, nodeNumberOfEachCellRe + colorGroupStart + colorGroupID);
				zmmQNS = _mm512_loadu_pd(tCellRe[0]+ colorGroupStart + colorGroupID);
				for (int offsetNode = 1; offsetNode <= maxNodes; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(0xFF, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(0xFF, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
					zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(0xFF, zmmNodePosi, zmmOffReal);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, cell2Node, 4);
					zmmAccessFrequency = _mm512_i32gather_pd(zmmPosiReal, cell2NodeCountRe, 8);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, tNodeAVX[0], 8);
					zmmFmad = _mm512_mask3_fmadd_pd(zmmQNS, zmmAccessFrequency, zmmQNode, kZero);
					_mm512_i32scatter_pd(tNodeAVX[0], zmmNodeID, zmmFmad, 8);
					zmmNCount = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmNodeID, nCountAVX, 4);
					zmmAccessInt = _mm256_mmask_i32gather_epi32(zmmIndexZero, 0xFF, zmmPosiReal, cell2NodeCountIntRe, 4);
					zmmNCount = _mm256_mask_add_epi32(zmmNCount, kZero, zmmAccessInt, zmmNCount);
					_mm256_i32scatter_epi32(nCountAVX, zmmNodeID, zmmNCount, 4);
				}
			}
			if (n_tail > 0){
				kNTail = 0xFF>>(8-n_tail);
				zmmNodePosi = _mm256_maskz_loadu_epi32(kNTail, cell2NodePositionRe + colorGroupStart + n_block);
				zmmNodeNumberOfEachFace = _mm256_maskz_loadu_epi32(kNTail, nodeNumberOfEachCellRe + colorGroupStart + n_block);
				zmmQNS = _mm512_maskz_loadu_pd(kNTail, tCellRe[0]+ colorGroupStart + n_block);
				for (int offsetNode = 1; offsetNode <= maxNodes; offsetNode++){
					zmmNum = _mm256_maskz_set1_epi32(kNTail, offsetNode);
					zmmOffset = _mm256_maskz_sub_epi32(kNTail, zmmNodeNumberOfEachFace, zmmNum);
					kZero = _mm256_cmp_epi32_mask(zmmMinusOne, zmmOffset, _MM_CMPINT_LT);
					zmmOffReal = _mm256_mask_blend_epi32(kZero, zmmIndexZero, zmmOffset);
					zmmPosiReal = _mm256_maskz_add_epi32(kNTail, zmmNodePosi, zmmOffReal);
					zmmNodeID = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmPosiReal, cell2Node, 4);
					zmmAccessFrequency = _mm512_i32gather_pd(zmmPosiReal, cell2NodeCountRe, 8);
					zmmQNode = _mm512_i32gather_pd(zmmNodeID, tNodeAVX[0], 8);
					zmmFmad = _mm512_mask3_fmadd_pd(zmmQNS, zmmAccessFrequency, zmmQNode, kZero);
					_mm512_mask_i32scatter_pd(tNodeAVX[0], kNTail, zmmNodeID, zmmFmad, 8);
					zmmNCount = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmNodeID, nCountAVX, 4);
					zmmAccessInt = _mm256_mmask_i32gather_epi32(zmmIndexZero, kNTail, zmmPosiReal, cell2NodeCountIntRe, 4);
					zmmNCount = _mm256_mask_add_epi32(zmmNCount, kZero, zmmAccessInt, zmmNCount);
					_mm256_mask_i32scatter_epi32(nCountAVX, kNTail, zmmNodeID, zmmNCount, 4);

				}
			}
		}
	TIMERCPU1("AVX512CellLoopSepNCQTNodeSep");

	//validate
   if (loopID == 0){
	for (int nodeID = 0; nodeID < nTotalNode; nodeID++){
		for (int equationID = 0; equationID < nEquation; equationID++){
			if (abs(qNodeAVX[equationID][nodeID] - qNode[equationID][nodeID])>1.0e-12)
			{
				printf("Error: nodeID = %d, equationID = %d, qNodeAVX = %.30e, qNode = %.30e\n", nodeID, equationID, qNodeAVX[equationID][nodeID], qNode[equationID][nodeID]);
				exit(1);
			
			}
		}

		if (abs(tNodeAVX[0][nodeID] - tNode[0][nodeID])>1.0e-12){
				printf("Error: nodeID = %d, tNodeAVX = %.30e, tNode = %.30e\n", nodeID, tNodeAVX[0][nodeID], tNode[0][nodeID]);
				exit(1);
			
		}
		if (abs(nCountAVX[nodeID] - nCount[nodeID])>1.0e-12){
				printf("Error: nodeID = %d, nCountAVX = %d, nCount = %d\n", nodeID, nCountAVX[nodeID], nCount[nodeID]);
				exit(1);
			
		}
	}
	printf("Test qNode tNode successfully\n");
   }

}
