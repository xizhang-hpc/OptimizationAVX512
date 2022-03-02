#include "stdio.h"
#include "HostInteriorFaceNCountQNodeTNodeCal.h"
#include "GlobalVariablesHost.h"
#include "constantVars.hxx"
#include "newTimer.h"
#include "stdlib.h"
#include "immintrin.h"
#define N_UNROLL 8
using namespace newTimer;
void initNCountQNodeTNodeZero(const int loopID){
	if (loopID == 0) printf("initNCountQNodeTNodeZero\n");
	if ((!tNode)||(!qNode)||(!nCount)) {
		printf("Error: tCell, qNode, and nCount should be allodated memory\n");
		exit(0);
	}
	int nEquation = nl + nchem;
	int iNode;
	int iEquation;
	for (iNode = 0; iNode < nTotalNode; iNode++){
		for (iEquation = 0; iEquation < nEquation; iEquation++){
			qNode[iEquation][iNode] = 0.0;
		}
		for (iEquation = 0; iEquation < nTemperature; iEquation++){
			tNode[iEquation][iNode] = 0.0;
		}
		nCount[iNode] = 0;
	}
}
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
void HostAverageQNodeTNode(const int loopID){
	if (loopID == 0) printf("HostAverageQNodeTNode\n");
	int nEquation = nl + nchem;
	int iNode;
	int iEquation;
	for (iNode = 0; iNode < nTotalNode; iNode++){
		for (iEquation = 0; iEquation < nEquation; iEquation++){
			qNode[iEquation][iNode] /= nCount[iNode];
		}
		for (iEquation = 0; iEquation < nTemperature; iEquation++){
			tNode[iEquation][iNode] /= nCount[iNode];
		}
	}
}

void CallHostInteriorFaceNCountQNodeTNodeCal(const int loopID){
	if (loopID == 0) printf("HostInteriorFaceNCountQNodeTNodeCal\n");
	int le, re;
	int iFace;
	int nEquation = nl + nchem;		
    	for (iFace = nBoundFace; iFace < nTotalFace; ++ iFace) {
	        le = leftCellofFace[iFace];
        	re = rightCellofFace[iFace];
		int jNode;
		int nodePosition = face2NodePosition[iFace];
	        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        	// From left
        		int point = face2Node[nodePosition + jNode];
			//add for test
			//if (point == 72) printf("iFace = %d, jNode = %d, le = %d, re = %d\n", iFace, jNode, le, re);
			//add ends
            		for (int m = 0; m < nEquation; m++) {
	        		qNode[m][point] += qNS[m][le];
        		}            
            		tNode[0][point] += tCell[0][le];
            		nCount[point] += 1;

            		for (int m = 0; m < nEquation; ++ m) {
                		qNode[m][point] += qNS[m][re];
            		}
            		tNode[0][point] += tCell[0][re];
            		nCount[point] += 1;
		}
    	}

}

void CallHostFaceNCountQNodeTNodeCal(const int loopID){
	if (loopID == 0) printf("HostFaceNCountQNodeTNodeCal\n");
	//initialization: set nCount, qNode, tNode as zero
	initNCountQNodeTNodeZero(loopID);
	int le, re;
	int iFace;
	int nEquation = nl + nchem;		
	TIMERCPU0("HostFaceLoopNCQTNodeWhole", "face loop for NCQTNode");
/*
    	for (iFace = 0; iFace < nTotalFace; ++ iFace) {
	        le = leftCellofFace[iFace];
        	re = rightCellofFace[iFace];
		int jNode;
		int nodePosition = face2NodePosition[iFace];
	        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        	// From left
        		int point = face2Node[nodePosition + jNode];
            		for (int m = 0; m < nEquation; m++) {
	        		qNode[m][point] += qNS[m][le];
        		}            
            		tNode[0][point] += tCell[0][le];
            		nCount[point] += 1;
			//do not consider ghost cells
			if (re < nTotalCell) {
            			for (int m = 0; m < nEquation; ++ m) {
                			qNode[m][point] += qNS[m][re];
            			}
	            		tNode[0][point] += tCell[0][re];
        	    		nCount[point] += 1;
			}
		}
    	}
*/
    	for (iFace = 0; iFace < nBoundFace; ++ iFace) {
	        le = leftCellofFace[iFace];
		int jNode;
		int nodePosition = face2NodePosition[iFace];
	        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        	// From left
        		int point = face2Node[nodePosition + jNode];
            		for (int m = 0; m < nEquation; m++) {
	        		qNode[m][point] += qNS[m][le];
        		}            
            		tNode[0][point] += tCell[0][le];
            		nCount[point] += 1;
		}
    	}
    	for (iFace = nBoundFace; iFace < nTotalFace; ++ iFace) {
	        le = leftCellofFace[iFace];
        	re = rightCellofFace[iFace];
		int jNode;
		int nodePosition = face2NodePosition[iFace];
	        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        	//From left
        		int point = face2Node[nodePosition + jNode];
            		for (int m = 0; m < nEquation; m++) {
	        		qNode[m][point] += qNS[m][le];
        		}            
            		tNode[0][point] += tCell[0][le];
            		nCount[point] += 1;
			//From Right
            		for (int m = 0; m < nEquation; ++ m) {
                		qNode[m][point] += qNS[m][re];
            		}
	            	tNode[0][point] += tCell[0][re];
        	    	nCount[point] += 1;
		}
    	}
	TIMERCPU1("HostFaceLoopNCQTNodeWhole");

	initNCountQNodeTNodeZero(loopID);
	TIMERCPU0("HostFaceLoopNCQTNodeSep", "Separate face loop for NCQTNode");
        for (int m = 0; m < nEquation; m++) {
    		for (iFace = 0; iFace < nBoundFace; ++ iFace) {
	        	le = leftCellofFace[iFace];
			int jNode;
			int nodePosition = face2NodePosition[iFace];
	        	for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        		// From left
	        		int point = face2Node[nodePosition + jNode];
	        		qNode[m][point] += qNS[m][le];
        		}            
		}
    	}
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
	__m256i zmmIndexZero = _mm256_maskz_set1_epi32(0xFF, 0);
	__m256i zmmMinusOne = _mm256_maskz_set1_epi32(0xFF, -1);
	__m512d zmmZero = _mm512_setzero_pd();

	__m512d zmmQNS;
	__m512d zmmQNSRight;
	__m512d zmmQNode;
	__m512d zmmAdd;

	__mmask8 kZero;
	initNCountQNodeTNodeAVXZero(loopID);
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
		for (int faceID = 0; faceID < nBoundFace; faceID++){
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
	printf("Test AVX512 of qNode on boundary successfully\n");
        for (int m = 0; m < nEquation; ++ m) {
    		for (iFace = nBoundFace; iFace < nTotalFace; ++ iFace) {
	        	le = leftCellofFace[iFace];
	        	re = rightCellofFace[iFace];
			int jNode;
			int nodePosition = face2NodePosition[iFace];
		        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
		        	//From left
        			int point = face2Node[nodePosition + jNode];
	        		qNode[m][point] += qNS[m][le];
				//From Right
                		qNode[m][point] += qNS[m][re];
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
	printf("Test AVX512 of qNode on the whole face successfully\n");
    	for (iFace = 0; iFace < nBoundFace; ++ iFace) {
	        le = leftCellofFace[iFace];
		int jNode;
		int nodePosition = face2NodePosition[iFace];
	        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        	// From left
        		int point = face2Node[nodePosition + jNode];
            		tNode[0][point] += tCell[0][le];
            		nCount[point] += 1;
		}
    	}
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
				}
				

			}
		}
		for (int faceID = 0; faceID < nBoundFace; faceID++){
			int nodeNum = nodeNumberOfEachFace[faceID];
			int nodePosi = face2NodePosition[faceID];
			for (int offset = 0; offset < nodeNum; offset++){
				int nodeID = face2Node[nodePosi + offset];
				if (abs(tNodeAVX[0][nodeID] - tNode[0][nodeID]) >  1.0e-12) {
					printf("Error: nBoundFace = %d, faceID = %d, nodeID = %d, tNodeAVX = %.30e, tNode = %.30e\n", nBoundFace, faceID, nodeID, tNodeAVX[0][nodeID], tNode[0][nodeID]);
					exit(0);
				}
			}
		}
		printf("Test AVX512 of tNode on boundary successfully\n");
    	for (iFace = nBoundFace; iFace < nTotalFace; ++ iFace) {
	        le = leftCellofFace[iFace];
        	re = rightCellofFace[iFace];
		int jNode;
		int nodePosition = face2NodePosition[iFace];
	        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        	//From left
        		int point = face2Node[nodePosition + jNode];
            		tNode[0][point] += tCell[0][le];
            		nCount[point] += 1;
			//From Right
	            	tNode[0][point] += tCell[0][re];
        	    	nCount[point] += 1;
		}
    	}
	TIMERCPU1("HostFaceLoopNCQTNodeSep");
	//Get the average qNode and tNode by dividing nCount
	HostAverageQNodeTNode(loopID);
}

void CallHostNodeLoopNCountQNodeTNodeCal(const int loopID){
	if (loopID == 0) printf("HostNodeLoopNCountQNodeTNodeCal\n");
	//initialization: set nCount, qNode, tNode as zero
	initNCountQNodeTNodeZero(loopID);
	int cellPosition, numCellsInNode, cellOffset, cellID;
	int nodeID;
	int nEquation = nl + nchem;		
	TIMERCPU0("HostNodeLoopNCQTNode", "node loop for NCQTNode");
    	for (nodeID = 0; nodeID < nTotalNode; nodeID++) {
		cellPosition = node2CellPosition[nodeID];
		numCellsInNode = cellNumberOfEachNode[nodeID];
		for (cellOffset = 0; cellOffset < numCellsInNode; cellOffset++){
			cellID = node2Cell[cellPosition + cellOffset];
			//add for test
				//if ((nodeID == 44512)&&(loopID == 0)) printf("nodeID = %d, cellID = %d, tCell = %.30e\n", nodeID, cellID, tCell[0][cellID]);
			//add end
			if (cellID < nTotalCell) {
            			for (int m = 0; m < nEquation; m++) {
	        			qNode[m][nodeID] += qNS[m][cellID];
        			}            
            			tNode[0][nodeID] += tCell[0][cellID];
            			nCount[nodeID] += 1;
			}
		}
    	}
	TIMERCPU1("HostNodeLoopNCQTNode");
	//Get the average qNode and tNode by dividing nCount
	HostAverageQNodeTNode(loopID);
}

void CallHostNodeLoopNCountQNodeTNodeCalModify(const int loopID){
	if (loopID == 0) printf("HostNodeLoopNCountQNodeTNodeCalModify\n");
	//initialization: set nCount, qNode, tNode as zero
	initNCountQNodeTNodeZero(loopID);
	int cellPosition, numCellsInNode, cellOffset, cellID;
	int nodeID;
	int nEquation = nl + nchem;		
	TIMERCPU0("HostNodeLoopNCQTNodeModify", "node loop for NCQTNode with modification of nCount");
    	for (nodeID = 0; nodeID < nTotalNode; nodeID++) {
		cellPosition = node2CellPosition[nodeID];
		numCellsInNode = cellNumberOfEachNode[nodeID];
		for (cellOffset = 0; cellOffset < numCellsInNode; cellOffset++){
			cellID = node2Cell[cellPosition + cellOffset];
			//add for test
				//if ((nodeID == 44512)&&(loopID == 0)) printf("nodeID = %d, cellID = %d, tCell = %.30e\n", nodeID, cellID, tCell[0][cellID]);
			//add end
			if (cellID < nTotalCell) {
            			for (int m = 0; m < nEquation; m++) {
	        			qNode[m][nodeID] += qNS[m][cellID] * node2CellCount[cellPosition + cellOffset];
        			}            
            			tNode[0][nodeID] += tCell[0][cellID] * node2CellCount[cellPosition + cellOffset];
            			nCount[nodeID] += 1 * node2CellCount[cellPosition + cellOffset];
			}
		}
    	}
	TIMERCPU1("HostNodeLoopNCQTNodeModify");
	//Get the average qNode and tNode by dividing nCount
	HostAverageQNodeTNode(loopID);
}

void CallHostNodeLoopNCountQNodeTNodeCalFinal(const int loopID){
	if (loopID == 0) printf("HostNodeLoopNCountQNodeTNodeCalFinal\n");
	//initialization: set nCount, qNode, tNode as zero
	initNCountQNodeTNodeZero(loopID);
	int cellPosition, numCellsInNode, cellOffset, cellID;
	int nodeID;
	int nEquation = nl + nchem;		
	TIMERCPU0("HostNodeLoopNCQTNodeFinal", "node loop for NCQTNode with computing nCount");
    	for (nodeID = 0; nodeID < nTotalNode; nodeID++) {
		cellPosition = node2CellPosition[nodeID];
		numCellsInNode = cellNumberOfEachNode[nodeID];
		for (cellOffset = 0; cellOffset < numCellsInNode; cellOffset++){
			cellID = node2Cell[cellPosition + cellOffset];
			//add for test
				//if ((nodeID == 44512)&&(loopID == 0)) printf("nodeID = %d, cellID = %d, tCell = %.30e\n", nodeID, cellID, tCell[0][cellID]);
			//add end
            		for (int m = 0; m < nEquation; m++) {
	        		qNode[m][nodeID] += qNS[m][cellID] * node2CellCount[cellPosition + cellOffset];
        		}            
            		tNode[0][nodeID] += tCell[0][cellID] * node2CellCount[cellPosition + cellOffset];
            		nCount[nodeID] += 1 * node2CellCount[cellPosition + cellOffset];
		}
    	}
	TIMERCPU1("HostNodeLoopNCQTNodeFinal");
	//Get the average qNode and tNode by dividing nCount
	HostAverageQNodeTNode(loopID);
}

void CallHostCellLoopNCountQNodeTNodeCal(const int loopID){
	if (loopID == 0) printf("HostCellLoopNCountQNodeTNodeCal\n");
	//initialization: set nCount, qNode, tNode as zero
	initNCountQNodeTNodeZero(loopID);
	int nodePosition, numNodesInCell, nodeOffset, cellID;
	int nodeID;
	int nEquation = nl + nchem;		
	TIMERCPU0("HostCellLoopNCQTNode", "cell loop for NCQTNode");
    	for (cellID = 0; cellID < nTotalCell; cellID++) {
		nodePosition = cell2NodePosition[cellID];
		numNodesInCell = nodeNumberOfEachCell[cellID];
		for (nodeOffset = 0; nodeOffset < numNodesInCell; nodeOffset++){
			nodeID = cell2Node[nodePosition + nodeOffset];
            		for (int m = 0; m < nEquation; m++) {
	        		qNode[m][nodeID] += qNS[m][cellID];
        		}            
            		tNode[0][nodeID] += tCell[0][cellID];
            		nCount[nodeID] += 1;
		}
    	}
	TIMERCPU1("HostCellLoopNCQTNode");
	//Get the average qNode and tNode by dividing nCount
	HostAverageQNodeTNode(loopID);
}

void CallHostCellLoopNCountQNodeTNodeCalFinal(const int loopID){
	if (loopID == 0) printf("HostCellLoopNCountQNodeTNodeCalFinal\n");
	//initialization: set nCount, qNode, tNode as zero
	initNCountQNodeTNodeZero(loopID);
	int nodePosition, numNodesInCell, nodeOffset, cellID;
	int nodeID;
	int accessFrequency;
	int nEquation = nl + nchem;		
	TIMERCPU0("HostCellLoopNCQTNodeFinal", "cell loop for NCQTNode with computing nCount");
    	for (cellID = 0; cellID < nTotalCell; cellID++) {
		nodePosition = cell2NodePosition[cellID];
		numNodesInCell = nodeNumberOfEachCell[cellID];
		for (nodeOffset = 0; nodeOffset < numNodesInCell; nodeOffset++){
			nodeID = cell2Node[nodePosition + nodeOffset];
			accessFrequency = cell2NodeCount[nodePosition + nodeOffset];

            		for (int m = 0; m < nEquation; m++) {
	        		qNode[m][nodeID] += qNS[m][cellID] * accessFrequency;
        		}            
            		tNode[0][nodeID] += tCell[0][cellID] * accessFrequency;
            		nCount[nodeID] += 1 * accessFrequency;
		}
    	}
	TIMERCPU1("HostCellLoopNCQTNodeFinal");
	//Get the average qNode and tNode by dividing nCount
	HostAverageQNodeTNode(loopID);
}
