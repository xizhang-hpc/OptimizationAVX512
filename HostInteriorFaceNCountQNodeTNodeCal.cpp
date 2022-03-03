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
	//HostAverageQNodeTNode(loopID);
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
