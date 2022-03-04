#include "stdio.h"
#include "GlobalVariablesHost.h"
#include "stdlib.h"
#include <vector>
#include <algorithm>
using namespace std;
int * bcType; //boundary type array
int * leftCellofFace; //Topology variable owner
int * rightCellofFace;//Topology variable neighbour
int * leftCellofFaceRe; //reorder topology variable owner by face coloring
int * rightCellofFaceRe;//reorder topology variable neighbour by face coloring
int * leftCellofFaceNodeRe; //reorder leftCellofFace by face coloring according to nodes
int * rightCellofFaceNodeRe;//reorder
int ** cell2Face;
int ** cell2Cell;
int ** cell2FaceRe;
int ** cell2CellRe;
int * faceNumberOfEachCell;
int * leftFaceNumberOfEachCell;
int * cell2FacePosition;
int ** leftRightFace;
int * face2Node; //Topology information, face's nodes
int * face2NodeNodeRe; //face coloring according to node
int * nodeNumberOfEachFace; //Topology information, node number owned by a face
int * nodeNumberOfEachFaceNodeRe; //face coloring according to node
int * face2NodePosition; //The start position of one face in face2Node
int * face2NodePositionNodeRe; //face coloring according to node
int * node2Cell; //Topology information, node's cells
int * node2CellCount; //The cell-access frequency by a node.
int * cellNumberOfEachNode; //Topology information, cell number owned by a node
int * node2CellPosition; 
int * cell2Node; //Topology information, cell's nodes
int * cell2NodeCount; //The cell-access frequency by a node, sorted by cell.
fpkind * cell2NodeCountRe; //The cell-access frequency by a node, sorted by cell.
int * nodeNumberOfEachCell; //Topology information, node number owned by a cell
int * nodeNumberOfEachCellRe; //Topology information, node number owned by a cell
int * cell2NodePosition; //The start position of one cell in cell2Node
int * cell2NodePositionRe; //The start position of one cell in cell2Node
int * node2Face;
int * faceNumberOfEachNode;
int * node2FacePosition;
int * isGhostCell; 
int * BoundFaceGroup;
int * BoundFaceGroupPosi;
int * BoundFaceGroupNum;
int BoundFaceColorNum = 0;
int * InteriorFaceGroup;
int * InteriorFaceGroupPosi;
int * InteriorFaceGroupNum;
int InteriorFaceColorNum = 0;
int * BoundFaceNodeGroup; //Coloring faces by nodes
int * BoundFaceNodeGroupPosi; //The offset of the first face label of one face group in BoundFaceNodeGroup
int * BoundFaceNodeGroupNum; //the number of every face group (the number of faces in one color) according to node
int BoundFaceNodeColorNum = 0; //the total number of color (the total number of face groups)
int * InteriorFaceNodeGroup;
int * InteriorFaceNodeGroupPosi;
int * InteriorFaceNodeGroupNum;
int InteriorFaceNodeColorNum = 0;
int * CellNodeGroup; //cell coloring by nodes. Cells in a color group will not share the same node.
int * CellNodeGroupPosi;
int * CellNodeGroupNum;
int CellNodeColorNum = 0;
double * xfn;
double * yfn;
double * zfn;
double * vgn;
double * xtn;
double * ytn;
double * ztn;
double * area;
double * xfc;
double * yfc;
double * zfc;
double * xcc;
double * ycc;
double * zcc;
fpkind ** qNS;
fpkind ** qNSRe;
fpkind ** qNSL;
fpkind ** qNSR;
fpkind ** qNSLOrg;
fpkind ** qNSROrg;
fpkind ** qNSLRFaceCpy;
fpkind ** dqdx;
fpkind ** dqdy;
fpkind ** dqdz;
fpkind ** tCell; //temperature on cells
fpkind ** res; //residual for ns equations
fpkind ** resOrg;
fpkind ** resAVX; //residual by AVX512
fpkind ** flux; //flux for 5 unkonwns rho, u, v, w, p on faces.
fpkind ** fluxRe; //reorder flux by face coloring
fpkind ** qNode;
fpkind ** qNodeAVX;
fpkind ** tNode;
fpkind ** tNodeAVX;
fpkind * limit;
int * nCount; //cell number of each node by calculation. what's the difference between nCount and nodeNumberOfEachCell (nCPN)??
int * nCountAVX; //cell number of each node by calculation. what's the difference between nCount and nodeNumberOfEachCell (nCPN)??
fpkind * dMin; //local minimum value of q on cells, by calculation
fpkind * dMax; //local maxmum value of q on cells, by calculation
fpkind * dMinAVX; //AVX
fpkind * dMaxAVX; //AVX
void setPropertyOfSimulation(){
	nl = 5; //rho, u, v, w, p
	nchem = 0; //do not consider chemistry
	nTemperature = 1; 
	//SEG_LEN = 1000;
}
void setSEGLEN(){
	if (SEG_LEN != 0) return;
	if (nTotalFace == 0){
		printf("nTotalFace should be read from owner or neighbour firstly\n");
		exit(0);
	}
	SEG_LEN = nTotalFace;
}
void readLeftRightCellofFace(){
	//judge whether or not leftCellofFace or rightCellofFace exists
	if ((leftCellofFace)||(rightCellofFace)) return;
        FILE * fileLeftCellOfFace;
        FILE * fileRightCellOfFace;	
        fileLeftCellOfFace = fopen("leftCellOfFace", "r");
        fileRightCellOfFace = fopen("rightCellOfFace", "r");
        if ((fileLeftCellOfFace == NULL)||(fileRightCellOfFace == NULL)) {
        	printf("Error: fail to open files leftCellOfFace and rightCellOfFace\n");
                exit(0);
        }
	fread(&nTotalFace, sizeof(int), 1, fileLeftCellOfFace);	
	//alloc memory for leftCellofFace and rightCellofFace by nTotalFace
	leftCellofFace = (int *)malloc(nTotalFace * sizeof(int));
	rightCellofFace = (int *)malloc(nTotalFace * sizeof(int));
	fread(leftCellofFace, sizeof(int), nTotalFace, fileLeftCellOfFace);
	fread(&nTotalFace, sizeof(int), 1, fileRightCellOfFace);	
	fread(rightCellofFace, sizeof(int), nTotalFace, fileRightCellOfFace);
	fclose(fileLeftCellOfFace);
	fclose(fileRightCellOfFace);
}
void readBoundaryType(){
	if (bcType) return;
	FILE * boundaryTypeFile;
	boundaryTypeFile  = fopen("boundaryType", "r");
	if (boundaryTypeFile == NULL) {
		printf("Something wrong with the file boundaryType\n");
		exit(0);
	}
	fread(&nBoundFace, sizeof(int), 1, boundaryTypeFile);
	bcType = (int *)malloc(nBoundFace * sizeof(int));
	fread(bcType, sizeof(int), nBoundFace, boundaryTypeFile);
	fclose(boundaryTypeFile);
}

void readNumberFaceOfEaceCell(){
	if (faceNumberOfEachCell) return;
	FILE * numFaceOfEachCellFile;
	numFaceOfEachCellFile = fopen("numberFaceOfEachCell", "r");
	if (numFaceOfEachCellFile == NULL) {
		printf("Something wrong with file numberFaceOfEachCell\n");
		exit(0);
	}
	fread(&nTotalCell, sizeof(int), 1, numFaceOfEachCellFile);	
	faceNumberOfEachCell = (int *)malloc(sizeof(int) * nTotalCell);
	fread(faceNumberOfEachCell, sizeof(int), nTotalCell, numFaceOfEachCellFile);
	fclose(numFaceOfEachCellFile);
}

void readCell2Face(){
	if (cell2Face) return;
	printf("readCell2Face\n");
	//if (nTotalCell == 0){
	if (!faceNumberOfEachCell){
		printf("Read numberFaceOfEachCell firstly \n");
		exit(0);
	}
	FILE * cell2FaceFile;
	cell2FaceFile = fopen("cell2Face","r");
	if (cell2FaceFile == NULL){
		printf("something wrong with file cell2Face\n");
		exit(0);
	}
	//allocate memory of cell2Face
	cell2Face = (int **)malloc(sizeof(int *)*nTotalCell);
	//allocate memory of cell2FacePosition
	cell2FacePosition = (int *)malloc(nTotalCell * sizeof(int));
	int cellID;
	int numOfFaces = 0;
	for (cellID = 0; cellID < nTotalCell; cellID++)	{
		cell2FacePosition[cellID] = numOfFaces;
		numOfFaces += faceNumberOfEachCell[cellID];
	}
	cell2Face[0] = (int *)malloc(sizeof(int)*numOfFaces);
	//read from cell2FaceFile into cell2Face
	fread(cell2Face[0], sizeof(int), numOfFaces, cell2FaceFile);
	//reset array mode of cell2Face
	for (cellID = 1; cellID < nTotalCell; cellID++){
		int numFaceInCellLast = faceNumberOfEachCell[cellID-1];
		cell2Face[cellID] = &cell2Face[cellID-1][numFaceInCellLast];
	}
	fclose(cell2FaceFile);
}

void setLeftRightFace(){
	if ((!leftCellofFace)||(!rightCellofFace)) {
		printf("Error: leftCellofFace or rightCellofFace should be read firstly\n");
		exit(0);
	}
	if ((!cell2Face)||(!faceNumberOfEachCell) ){
		printf("Error: cell2Face or faceNumberOfEachCell should be read firstly\n");
		exit(0);
	}
	printf("setLeftRightFace\n");
	int cellID, faceID;
	int numFaces = 0;
	//allocate leftRightFace memory
	for (cellID = 0; cellID < nTotalCell; cellID++){
		numFaces += faceNumberOfEachCell[cellID];
	}
	leftRightFace = (int **)malloc(nTotalCell * sizeof(int *));
	leftRightFace[0] = (int *)malloc(numFaces * sizeof(int));
	for (cellID = 1; cellID < nTotalCell; cellID++){
		int numFaceInCellLast = faceNumberOfEachCell[cellID-1];
		leftRightFace[cellID] = &leftRightFace[cellID-1][numFaceInCellLast];
	}
	int offsetFace;
	int le, re;
	for (cellID = 0; cellID < nTotalCell; cellID++){
		for (offsetFace = 0; offsetFace < faceNumberOfEachCell[cellID]; offsetFace++) {
			faceID = cell2Face[cellID][offsetFace];
			le = leftCellofFace[faceID];
			re = rightCellofFace[faceID];
			if (le == cellID) leftRightFace[cellID][offsetFace] = 1;
			else if (re == cellID) leftRightFace[cellID][offsetFace] = -1;
			else {
				printf("something wrong\n");
				exit(0);
			}
		}
	}
}


void setFaceNumberOfEachNode(){
	if (faceNumberOfEachNode) return;
	if (!face2Node) {
		printf("Error: face2Node should be read firstly\n");
		exit(0);
	}
	faceNumberOfEachNode = (int *)malloc(nTotalNode * sizeof(int));
	int nodeID, faceID;
	int numNodesInFaces, nodePosition, offsetNode;
	//initialize faceNumberOfEachNode as zero
	for (nodeID = 0; nodeID < nTotalNode; nodeID++){
		faceNumberOfEachNode[nodeID] = 0;
	}
	//compute faceNumberOfEachNode by face2Node and nodeNumberOfEachFace
	for (faceID = 0; faceID < nTotalFace; faceID++){
		numNodesInFaces = nodeNumberOfEachFace[faceID];
		nodePosition = face2NodePosition[faceID];
		for (offsetNode = 0; offsetNode < numNodesInFaces; offsetNode++){
			nodeID = face2Node[nodePosition + offsetNode];
			faceNumberOfEachNode[nodeID]++;
		}
	}
	
}
void setNode2Face(){
	if (node2Face) return;
	if (!face2Node) {
		printf("Error: face2Node should be read firstly\n");
		exit(0);
	}
	if (!faceNumberOfEachNode) {
		printf("Error: faceNumberOfEachNode should be set firstly");
		exit(0);
	}
	int nodeID, faceID;
	int numNodesInFaces, nodePosition, offsetNode;
	node2FacePosition = (int *)malloc(nTotalNode * sizeof(int));
	//compute node2FacePosition by faceNumberOfEachNode
	//allocate memory for node2Face
	int sumFaces = 0;
	for (nodeID = 0; nodeID < nTotalNode; nodeID++){
		node2FacePosition[nodeID] = sumFaces;
		sumFaces += faceNumberOfEachNode[nodeID];
	}
	node2Face = (int *)malloc(sumFaces * sizeof(int));
	//compute node2Face by face2Node and nodeNumberOfEachFace
	int facePosition, offsetFaceNode;
	int * offsetFace = (int *)malloc(nTotalNode * sizeof(int)); //recording offset in node2Face
	for (nodeID = 0; nodeID < nTotalNode; nodeID++){
		offsetFace[nodeID] = 0;
	}
	
	for (faceID = 0; faceID < nTotalFace; faceID++){
		numNodesInFaces = nodeNumberOfEachFace[faceID];
		nodePosition = face2NodePosition[faceID];
		for (offsetNode = 0; offsetNode < numNodesInFaces; offsetNode++){
			nodeID = face2Node[nodePosition + offsetNode];
			facePosition = node2FacePosition[nodeID];
			offsetFaceNode = offsetFace[nodeID];
			node2Face[facePosition+offsetFaceNode] = faceID;
			offsetFace[nodeID]++;
		}
	}
	//test the result
	for (nodeID = 0; nodeID < nTotalNode; nodeID++){
		if (faceNumberOfEachNode[nodeID] != (offsetFace[nodeID])) {
			printf("Error: faceNumberOfEachNode is not equal to offsetFace in iterm %d, faceNumber = %d, offsetFace = %d\n", nodeID, faceNumberOfEachNode[nodeID], offsetFace[nodeID]);
			exit(0);
		}
	}
}

void readNumberNodeOfEachFace(){
	if (nodeNumberOfEachFace) return;
	FILE * fileNumberNodeOfEachFace = fopen("numberNodeOfEachFace", "r");
	if (fileNumberNodeOfEachFace == NULL) {
		printf("Error: file numberNodeOfEachFace cannot be read\n");
		exit(0);
	}	
	fread(&nTotalFace,sizeof(int), 1, fileNumberNodeOfEachFace);
	nodeNumberOfEachFace = (int *)malloc(nTotalFace * sizeof(int));
	fread(nodeNumberOfEachFace, sizeof(int), nTotalFace, fileNumberNodeOfEachFace);
	fclose(fileNumberNodeOfEachFace);
	//Get the maximum node number in a face
	int maxNodeNumber = 0;
	for (int faceID = 0; faceID < nTotalFace; faceID++){
		if (maxNodeNumber < nodeNumberOfEachFace[faceID]) maxNodeNumber = nodeNumberOfEachFace[faceID];
	}
	printf("the maximum node number in a face is %d\n", maxNodeNumber);
}

void readFace2Node(){
	if (face2Node) return;
	if (!nodeNumberOfEachFace) {
		printf("Error: read numberNodeOfEachFace firstly\n");
		exit(0);
	}
	FILE * fileFace2Node = fopen("face2Node", "r");
	if (fileFace2Node == NULL){
		printf("Error: file face2Node cannot be read\n");
		exit(0);
	}
	int numNodes = 0;
	int iFace;
	face2NodePosition = (int *)malloc(sizeof(int) * nTotalFace);
	for (iFace = 0; iFace < nTotalFace; iFace++){
		face2NodePosition[iFace] = numNodes;
		numNodes += nodeNumberOfEachFace[iFace];
	}
	face2Node = (int *)malloc(sizeof(int) * numNodes);
	fread(face2Node, sizeof(int), numNodes, fileFace2Node);
	fclose(fileFace2Node);

}

void readNumberCellOfEachNode(){
	if (cellNumberOfEachNode) return;
	FILE * fileNumberCellOfEachNode = fopen("numberCellOfEachNode", "r");
	if (fileNumberCellOfEachNode == NULL) {
		printf("Error: numberCellOfEachNode cannot be read \n");
		exit(0);
	}
	fread(&nTotalNode, sizeof(int), 1, fileNumberCellOfEachNode);	
	cellNumberOfEachNode = (int *)malloc(sizeof(int)*nTotalNode);
	fread(cellNumberOfEachNode, sizeof(int), nTotalNode, fileNumberCellOfEachNode);
	fclose(fileNumberCellOfEachNode);
}

void readNode2Cell(){
	if (node2Cell) return;
	if (!cellNumberOfEachNode) {
		printf("Error: file numberCellOfEachNode should be read firstly\n");
		exit(0);
	}
	FILE * fileNode2Cell = fopen("node2Cell","r");
	if (fileNode2Cell == NULL) {
		printf("Error: file node2Cell cannot be read\n");
	}
	node2CellPosition = (int *)malloc(sizeof(int) * nTotalNode); 
	int numCells = 0;
	int iNode;
	for (iNode = 0; iNode < nTotalNode; iNode++){
		node2CellPosition[iNode] = numCells;
		numCells += cellNumberOfEachNode[iNode];
	}
	node2Cell = (int *)malloc(sizeof(int) * numCells);
	fread(node2Cell, sizeof(int), numCells, fileNode2Cell);
	fclose(fileNode2Cell);
	
}

void readNumberNodeOfEachCell(){
	if (nodeNumberOfEachCell) return;
	FILE * fileNumberNodeOfEachCell = fopen("numberNodeOfEachCell","r");
	if (fileNumberNodeOfEachCell == NULL){
		exit(0);
	}
	fread(&nTotalCell, sizeof(int), 1, fileNumberNodeOfEachCell);
	nodeNumberOfEachCell = (int *)malloc(sizeof(int) * nTotalCell);
	fread(nodeNumberOfEachCell, sizeof(int), nTotalCell, fileNumberNodeOfEachCell);
	int maxNodes = 0;
	for (int cellID = 0; cellID < nTotalCell; cellID++){
		if (maxNodes < nodeNumberOfEachCell[cellID]) maxNodes = nodeNumberOfEachCell[cellID];
	}
	printf("maxNodes = %d\n", maxNodes);
	fclose(fileNumberNodeOfEachCell);
}

void readCell2Node(){
	if (cell2Node) return;
	if (!nodeNumberOfEachCell) {
		printf("Error: numberNodeOfEachCell should be read firstly\n");
		exit(0);
	}
	FILE * fileCell2Node = fopen("cell2Node","r");	
	cell2NodePosition = (int *)malloc(sizeof(int)*nTotalCell);
	int numNodes = 0;
	int iCell;
	for (iCell = 0; iCell < nTotalCell; iCell++){
		cell2NodePosition[iCell] = numNodes;
		numNodes += nodeNumberOfEachCell[iCell];
	}
	cell2Node = (int *)malloc(sizeof(int)*numNodes);
	fread(cell2Node, sizeof(int), numNodes, fileCell2Node);
	fclose(fileCell2Node);

}

void setCell2Cell(){
	if ((!cell2Face)||(!leftCellofFace)||(!rightCellofFace)||(!faceNumberOfEachCell)) {
		printf("cell2Face, leftFaceofCell, rightFaceofCell should be read\n");
		exit(0);
	}
	int cellID, faceID, re, le;
	int numFacesInCell, offsetFace;

	int numOfFaces = 0;
	for (cellID = 0; cellID < nTotalCell; cellID++)	{
		numOfFaces += faceNumberOfEachCell[cellID];
	}
	size_t sizeFace = sizeof(int) * numOfFaces;
	cell2Cell = (int **)malloc(nTotalCell * sizeof(int *));
	cell2Cell[0] = (int *)malloc(sizeFace * sizeof(int));
	for (cellID = 1; cellID < nTotalCell; cellID++){
		numFacesInCell = faceNumberOfEachCell[cellID-1];
		cell2Cell[cellID] = &cell2Cell[cellID-1][numFacesInCell];
	}
	for (cellID = 0; cellID < nTotalCell; cellID ++){
		numFacesInCell = faceNumberOfEachCell[cellID];
		for (offsetFace = 0; offsetFace < numFacesInCell; offsetFace++){
			faceID = cell2Face[cellID][offsetFace];
			re = rightCellofFace[faceID];
			le = leftCellofFace[faceID];
			if (re == cellID) cell2Cell[cellID][offsetFace] = le;
			else cell2Cell[cellID][offsetFace] = re;
		}
	}

}

void setCell2FaceRe(){
	if (cell2FaceRe) return;
	if (!cell2Face) {
		printf("Error: cell2Face should be set firstly\n");
		exit(0);
	}
	int cellID, faceID, offsetFace, le;	
	int offsetLeft, offsetRight;
	int numOfFaces = 0;
	int numFacesInCell, leftNumber;
	size_t sizeCell = sizeof(int) * nTotalCell;
	leftFaceNumberOfEachCell = (int *)malloc(sizeCell);
	for (cellID = 0; cellID < nTotalCell; cellID++)	{
		numOfFaces += faceNumberOfEachCell[cellID];
	}
	size_t sizeFace = sizeof(int) * numOfFaces;
	cell2FaceRe = (int **)malloc(nTotalCell * sizeof(int *));
	cell2FaceRe[0] = (int *)malloc(sizeFace * sizeof(int));
	for (cellID = 1; cellID < nTotalCell; cellID++){
		numFacesInCell = faceNumberOfEachCell[cellID-1];
		cell2FaceRe[cellID] = &cell2FaceRe[cellID-1][numFacesInCell];
	}

	for (cellID = 0; cellID < nTotalCell; cellID++){
		numFacesInCell = faceNumberOfEachCell[cellID];
		leftNumber = 0;
		for (offsetFace = 0; offsetFace < numFacesInCell; offsetFace++){
			faceID = cell2Face[cellID][offsetFace];
			le = leftCellofFace[faceID];
			if (le == cellID) leftNumber++;
		}
		leftFaceNumberOfEachCell[cellID] = leftNumber;
		offsetLeft = 0;
		offsetRight = 0;
		for (offsetFace = 0; offsetFace < numFacesInCell; offsetFace++){
			faceID = cell2Face[cellID][offsetFace];
			le = leftCellofFace[faceID];
			if (le == cellID) {
				cell2FaceRe[cellID][0+offsetLeft] = faceID;
				offsetLeft++;
			}
			else{
				cell2FaceRe[cellID][leftNumber+offsetRight] = faceID;
				offsetRight++;
			}
		}
	}

}

void setCell2CellRe(){
	if ((!cell2FaceRe)||(!leftCellofFace)||(!rightCellofFace)||(!faceNumberOfEachCell)) {
		printf("cell2FaceRe, leftFaceofCell, rightFaceofCell should be read\n");
		exit(0);
	}
	int cellID, faceID, re, le;
	int numFacesInCell, offsetFace;

	int numOfFaces = 0;
	for (cellID = 0; cellID < nTotalCell; cellID++)	{
		numOfFaces += faceNumberOfEachCell[cellID];
	}
	size_t sizeFace = sizeof(int) * numOfFaces;
	cell2CellRe = (int **)malloc(nTotalCell * sizeof(int *));
	cell2CellRe[0] = (int *)malloc(sizeFace * sizeof(int));
	for (cellID = 1; cellID < nTotalCell; cellID++){
		numFacesInCell = faceNumberOfEachCell[cellID-1];
		cell2CellRe[cellID] = &cell2CellRe[cellID-1][numFacesInCell];
	}
	for (cellID = 0; cellID < nTotalCell; cellID ++){
		numFacesInCell = faceNumberOfEachCell[cellID];
		for (offsetFace = 0; offsetFace < numFacesInCell; offsetFace++){
			faceID = cell2FaceRe[cellID][offsetFace];
			re = rightCellofFace[faceID];
			le = leftCellofFace[faceID];
			if (re == cellID) cell2CellRe[cellID][offsetFace] = le;
			else cell2CellRe[cellID][offsetFace] = re;
		}
	}

}

void setXYZfnRandom(){
	//whether or not xfn, yfn, zfn has existed.
	if ((xfn)||(yfn)||(zfn)) return;
	if (nTotalFace == 0){
		printf("nTotalFace should be read from owner or neighbour\n");
		exit(0);
	}
	int faceID;
	xfn = (double *)malloc(sizeof(double) * nTotalFace);
	yfn = (double *)malloc(sizeof(double) * nTotalFace);
	zfn = (double *)malloc(sizeof(double) * nTotalFace);

	for(faceID = 0; faceID < nTotalFace; faceID++){
		xfn[faceID] = RANDOMNUMBER(0.0,1.0, nTotalFace);
		yfn[faceID] = RANDOMNUMBER(0.0,1.0, nTotalFace);
		zfn[faceID] = RANDOMNUMBER(0.0,1.0, nTotalFace);
	}
}

void setVgnRandom(){
	//whether or not vgn has existed.
	if (vgn) return;
	if (nTotalFace == 0){
		printf("nTotalFace should be read from owner or neighbour\n");
		exit(0);
	}
	int faceID;
	vgn = (double *)malloc(sizeof(double) * nTotalFace);
	for(faceID = 0; faceID < nTotalFace; faceID++){
		vgn[faceID] = RANDOMNUMBER(-5.0,5.0, nTotalFace);
	}
}

void setXYZtnRandom(){
	//whether or not xtn, ytn, ztn has existed.
	if ((xtn)||(ytn)||(ztn)) return;
	if (nTotalFace == 0){
		printf("nTotalFace should be read from owner or neighbour\n");
		exit(0);
	}
	int faceID;
	xtn = (double *)malloc(sizeof(double) * nTotalFace);
	ytn = (double *)malloc(sizeof(double) * nTotalFace);
	ztn = (double *)malloc(sizeof(double) * nTotalFace);

	for(faceID = 0; faceID < nTotalFace; faceID++){
		xtn[faceID] = RANDOMNUMBER(-5.0,5.0, nTotalFace);
		ytn[faceID] = RANDOMNUMBER(-5.0,5.0, nTotalFace);
		ztn[faceID] = RANDOMNUMBER(-5.0,5.0, nTotalFace);
	}
}

void setQNSRandom(){
	//Judge whether or not qNS exists!
	if (qNS) return;
	if (nl == 0) {
		printf("The number of equations nl should be set fistly\n");
		exit(0);
	}
	if (nBoundFace == 0) {
		printf("nBoundFace should be read from boundaryType fistly\n");
		exit(0);
	}
	if (nTotalCell == 0) {
		printf("nTotalCell should be read from fistly\n");
		exit(0);
	}
	int equationID, cellID;
	int nTotal = nTotalCell + nBoundFace;
	//allocate memory for qNS
	qNS = (fpkind **)malloc(sizeof(fpkind*)*nl);
	qNS[0] = (fpkind *)malloc(sizeof(fpkind)*nl*nTotal);
	for (equationID = 1; equationID < nl; equationID++){
		//qNS[equationID] = (double *)malloc(sizeof(double)*nTotal);
		//qNS[equationID] = qNS[0][equationID*nTotal];
		qNS[equationID] = &qNS[equationID-1][nTotal];
	}
	//set qNS by random
	for (equationID = 0; equationID < nl; equationID++){
		for (cellID = 0; cellID < nTotal; cellID++){
			qNS[equationID][cellID] = fpkind(RANDOMNUMBER(-5.0,5.0, nTotal));
		}
	}

}

void setTCellRandom(){
	if (tCell) return;
	if (nTotalCell == 0) {
		printf("Error: nTotalCell should be read from numberFaceOfEachCell or numberNodeOfEachCell firstly\n");
		exit(0);
	}
	if (nBoundFace == 0){
		printf("Error: nBoundFace should be read from boundaryType firstly\n");
		exit(0);
	}
	if (nTemperature == 0) {
		printf("Error: nTemperature should be set by setPropertyOfSimulation firstly\n");
		exit(0);
	}
	int nTotal = nTotalCell + nBoundFace;
	tCell = (fpkind **)malloc(sizeof(fpkind *) * nTemperature);
	tCell[0] = (fpkind *)malloc(sizeof(fpkind) * nTemperature * nTotal);
	int iTemp;
	for (iTemp = 1; iTemp < nTemperature; iTemp ++) tCell[iTemp] = &tCell[iTemp-1][nTotal];
	int iCell;
	for (iCell = 0; iCell < nTotal; iCell ++){
		for (iTemp = 0; iTemp < nTemperature; iTemp++){
			tCell[iTemp][iCell] = fpkind(RANDOMNUMBER(-5.0,5.0, nTotal));
		}
	}
}

void mallocQNode(){
	if (qNode) return;
	if (nTotalNode == 0) {
		printf("Error: nTotalNode should be read from numberCellOfEachNode\n");
		exit(0);
	}
	if (nl == 0) {
		printf("Error: nl should be set by setPropertyOfSimulation firstly\n");
		exit(0);
	}
	int nEquation = nl + nchem;
	qNode = (fpkind **)malloc(sizeof(fpkind *) * nEquation);
	qNode[0] = (fpkind *)malloc(sizeof(fpkind) * nEquation * nTotalNode);
	qNodeAVX = (fpkind **)malloc(sizeof(fpkind *) * nEquation);
	qNodeAVX[0] = (fpkind *)malloc(sizeof(fpkind) * nEquation * nTotalNode);
	int iEquation;
	for (iEquation = 1; iEquation < nEquation; iEquation++) qNode[iEquation] = &qNode[iEquation - 1][nTotalNode];
	for (iEquation = 1; iEquation < nEquation; iEquation++) qNodeAVX[iEquation] = &qNodeAVX[iEquation - 1][nTotalNode];
}

void setQNodeRandom(){
	if (!qNode) {
		printf("Error: qNode should be allocated memory firstly\n");
		exit(0);
	}
	int nEquation = nl + nchem;
	int equationID, nodeID;
	for (equationID = 0; equationID < nEquation; equationID++){
		for (nodeID = 0; nodeID < nTotalNode; nodeID++){
			qNode[equationID][nodeID]=RANDOMNUMBER(-2.0, 2.0, nTotalNode);
		}
	}
}

void mallocTNode(){
	if (tNode) return;
	if (nTotalNode == 0) {
		printf("Error: nTotalNode should be read from numberCellOfEachNode\n");
		exit(0);
	}
	if (nTemperature == 0) {
		printf("Error: nl should be set by setPropertyOfSimulation firstly\n");
		exit(0);
	}
	tNode = (fpkind **)malloc(sizeof(fpkind *) * nTemperature);
	tNode[0] = (fpkind *)malloc(sizeof(fpkind) * nTemperature * nTotalNode);
	tNodeAVX = (fpkind **)malloc(sizeof(fpkind *) * nTemperature);
	tNodeAVX[0] = (fpkind *)malloc(sizeof(fpkind) * nTemperature * nTotalNode);
	int iEquation;
	for (iEquation = 1; iEquation < nTemperature; iEquation++) tNode[iEquation] = &tNode[iEquation - 1][nTotalNode];
	for (iEquation = 1; iEquation < nTemperature; iEquation++) tNodeAVX[iEquation] = &tNodeAVX[iEquation - 1][nTotalNode];

}

void mallocNCount(){
	if (nCount) return;
	if (nTotalNode == 0) {
		printf("Error: nTotalNode should be read from numberCellOfEachNode\n");
		exit(0);
	}
	nCount = (int *)malloc(sizeof(int) * nTotalNode);
	nCountAVX = (int *)malloc(sizeof(int) * nTotalNode);
}

void setNode2CellCount(){
	if (node2CellCount) return;
	int le, re;
	int iFace;
	//Allocate memory and set zero for node2CellCount 
	int numCells = 0;
	int nodeID;
	for (nodeID = 0; nodeID < nTotalNode; nodeID++){
		numCells += cellNumberOfEachNode[nodeID];
	}
	size_t sizeCells = numCells * sizeof(int);
	node2CellCount = (int *)malloc(sizeCells);
	for (int i = 0; i < numCells; i++) node2CellCount[i] = 0;

    	for (iFace = 0; iFace < nTotalFace; ++ iFace) {
	        le = leftCellofFace[iFace];
        	re = rightCellofFace[iFace];
		int jNode;
		int nodePosition = face2NodePosition[iFace];
	        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        	// For node2CellCount
        		int point = face2Node[nodePosition + jNode];
			int start = node2CellPosition[point];
			int numCellsInNode = cellNumberOfEachNode[point];
			int offset, cellID;
			for (offset = 0; offset < numCellsInNode; offset++){
				cellID = node2Cell[start + offset];
				if (le == cellID) node2CellCount[start+offset]++;
				if ((re == cellID) && (re < nTotalCell)) node2CellCount[start+offset]++;
			}
		}

    	}

}

void setCell2NodeCount(){
	if (cell2NodeCount) return;
	int le, re;
	int iFace;
	//Allocate memory and set zero for cell2NodeCount
	int numNodes = 0;
	int cellID;
	for (cellID = 0; cellID < nTotalCell; cellID ++){
		numNodes += nodeNumberOfEachCell[cellID];
	}
	size_t sizeNodes = numNodes * sizeof(int);
	cell2NodeCount = (int *)malloc(sizeNodes);
	for (int i = 0; i < numNodes; i++) cell2NodeCount[i] = 0;

    	for (iFace = 0; iFace < nTotalFace; ++ iFace) {
	        le = leftCellofFace[iFace];
        	re = rightCellofFace[iFace];
		int jNode;
		int nodePosition = face2NodePosition[iFace];

	        for (jNode = 0; jNode < nodeNumberOfEachFace[iFace]; ++ jNode){
	        	// For cell2NodeCount	
        		int point = face2Node[nodePosition + jNode];
			// for left cell
			int start = cell2NodePosition[le];
			int numNodesInCell = nodeNumberOfEachCell[le];
			int offset, nodeID;
			for (offset = 0; offset < numNodesInCell; offset++){
				nodeID = cell2Node[start + offset];
				if (nodeID == point) cell2NodeCount[start+offset]++;
			}
			// for right cell
			if (re < nTotalCell) {
				int start = cell2NodePosition[re];
				int numNodesInCell = nodeNumberOfEachCell[re];
				int offset, nodeID;
				for (offset = 0; offset < numNodesInCell; offset++){
					nodeID = cell2Node[start + offset];
					if (nodeID == point) cell2NodeCount[start+offset]++;
				}
			}
		}
    	}

}

void setIsGhostCellHost(){
	if ((nTotalCell == 0)||(nBoundFace == 0)) {
		printf("Error: nTotalCell or nBoundFace should be read firstly\n ");
		exit(0);
	}
	int nTotal = nTotalCell + nBoundFace;
	isGhostCell = (int *)malloc(nTotal * sizeof(int));
	int cellID;
	for (cellID = 0; cellID < nTotal; cellID++) {
		if (cellID < nTotalCell) isGhostCell[cellID] = 0;
		else isGhostCell[cellID] = 1;
	}
}

void mallocQNSLR(){
	//judge whether or not qNSL and qNSR exists
	if ((qNSL) || (qNSR)) return;
	if (SEG_LEN == 0) {
		printf("SEG_LEN should be set firstly\n");
		exit(0);
	}
	if (nl == 0){
		printf("nl should be set firstly\n");
		exit(0);
	}
	int equationID, faceID;
	//allocate memory for qNSL and qNSR
	qNSL = (fpkind **)malloc(sizeof(fpkind *)*nl);
	qNSR = (fpkind **)malloc(sizeof(fpkind *)*nl);
	qNSL[0] = (fpkind *)malloc(sizeof(fpkind*)*nl*SEG_LEN);
	qNSR[0] = (fpkind *)malloc(sizeof(fpkind*)*nl*SEG_LEN);
	for (equationID = 1; equationID < nl; equationID++){
		//qNSL[equationID] = (double*)malloc(sizeof(double)*SEG_LEN);
		//qNSR[equationID] = (double*)malloc(sizeof(double)*SEG_LEN);
		//qNSL[equationID] = &qNSL[0][equationID*SEG_LEN];
		//qNSR[equationID] = &qNSR[0][equationID*SEG_LEN];
		qNSL[equationID] = &qNSL[equationID-1][SEG_LEN];
		qNSR[equationID] = &qNSR[equationID-1][SEG_LEN];
	}
	//Initialize qNSL and qNSR by random
	for (equationID = 0; equationID < nl; equationID++){
		for (faceID = 0; faceID < SEG_LEN; faceID++){
			qNSL[equationID][faceID] = 0.0;
			qNSR[equationID][faceID] = 0.0;
		}
	}
}

void mallocCpyQNSLROrg(){
	//judge whether or not qNSL and qNSR exists
	if ((qNSLOrg) || (qNSROrg)) return;
	if ((!qNSL)||(!qNSR)){
		printf("qNSL and qNSR should be allocated memory firstly\n");
		exit(0);
	}
	int equationID, faceID;
	//allocate memory for qNSL and qNSR
	qNSLOrg = (fpkind **)malloc(sizeof(fpkind *)*nl);
	qNSROrg = (fpkind **)malloc(sizeof(fpkind *)*nl);
	qNSLOrg[0] = (fpkind *)malloc(sizeof(fpkind*)*nl*SEG_LEN);
	qNSROrg[0] = (fpkind *)malloc(sizeof(fpkind*)*nl*SEG_LEN);
	for (equationID = 1; equationID < nl; equationID++){
		qNSLOrg[equationID] = &qNSLOrg[equationID-1][SEG_LEN];
		qNSROrg[equationID] = &qNSROrg[equationID-1][SEG_LEN];
	}
	//Initialize qNSL and qNSR by random
	for (equationID = 0; equationID < nl; equationID++){
		for (faceID = 0; faceID < SEG_LEN; faceID++){
			qNSLOrg[equationID][faceID] = qNSL[equationID][faceID];
			qNSROrg[equationID][faceID] = qNSR[equationID][faceID];
		}
	}
}

void mallocQNSLRFaceCpy(){
	if (qNSLRFaceCpy) return;
	if ((!qNSL)||(!qNSR)){
		printf("qNSL and qNSR should be allocated memory firstly\n");
		exit(0);
	}
	//allocate memory for qNSL and qNSR
	int nEquation = nl + nchem;
	int equationID, faceID;
	qNSLRFaceCpy = (fpkind **)malloc(nEquation * sizeof(fpkind *) );
	qNSLRFaceCpy[0] = (fpkind *)malloc(nEquation * SEG_LEN * 2 * sizeof(fpkind));
	for (equationID = 1; equationID < nEquation; equationID++){
		qNSLRFaceCpy[equationID] = &qNSLRFaceCpy[equationID-1][SEG_LEN * 2];
	}
	for (equationID = 0; equationID < nEquation; equationID++) {
		for (faceID = 0; faceID < SEG_LEN; faceID++) {
			qNSLRFaceCpy[equationID][2*faceID+0] = qNSL[equationID][faceID];
			qNSLRFaceCpy[equationID][2*faceID+1] = qNSR[equationID][faceID];
		}
	}
	
	
}

void setQNSLRRandom(){

	if ((!qNSL)||(!qNSR)){
		printf("Error: qNSL and qNSR should be allocated firstly\n");		
		exit(0);
	}
	int equationID, faceID;
	int nEquation = nl + nchem;

	for (equationID = 0; equationID < nEquation; equationID++){
		for (faceID = 0; faceID < SEG_LEN; faceID++){
			qNSL[equationID][faceID] = RANDOMNUMBER(-3.0, 4.0, SEG_LEN);
			qNSR[equationID][faceID] = RANDOMNUMBER(-3.0, 4.0, SEG_LEN);
		}
	}

}

void setAreaRandom(){
	if (area) return;
	if (nTotalFace == 0) {
		printf("Error: nTotalFace should be read firstly\n");
		exit(0);
	}
	area = (double *)malloc(sizeof(double)*nTotalFace);
	int areaI;
	for(areaI = 0; areaI < nTotalFace; areaI++){
		area[areaI] = RANDOMNUMBER(2.0, 4.0, nTotalFace);
	}

}

void setXYZfcRandom(){
	if (xfc) return;
	if (nTotalFace == 0){
		printf("Error: nTotalFace should be read firstly\n");
		exit(0);
	}
	int faceID;
	size_t sizeFace = nTotalFace * sizeof(double);
	xfc = (double *)malloc(sizeFace);
	yfc = (double *)malloc(sizeFace);
	zfc = (double *)malloc(sizeFace);
	for (faceID = 0; faceID < nTotalFace; faceID++){
		xfc[faceID] = RANDOMNUMBER(-1.0, 1.0, nTotalFace);
		yfc[faceID] = RANDOMNUMBER(-1.0, 1.0, nTotalFace);
		zfc[faceID] = RANDOMNUMBER(-1.0, 1.0, nTotalFace);
	}
}

void setXYZccRandom(){
	if (xcc) return;
	if ((nTotalCell == 0)||(nBoundFace == 0)) {
		printf("Error: nTotalCell or nBoundFace should be read firstly\n");
		exit(0);
	}
	int nTotal = nTotalCell + nBoundFace;
	size_t sizeCell = sizeof(double) * nTotal;
	xcc = (double *)malloc(sizeCell);
	ycc = (double *)malloc(sizeCell);
	zcc = (double *)malloc(sizeCell);
	int cellID;
	for (cellID = 0; cellID < nTotal; cellID++){
		xcc[cellID] = RANDOMNUMBER(-1.0, 1.0, nTotal);
		ycc[cellID] = RANDOMNUMBER(-2.0, 2.0, nTotal);
		zcc[cellID] = RANDOMNUMBER(-3.0, 3.0, nTotal);
	}
}

void mallocGradientQ(){
	if (dqdx) return;
	if (nl == 0) {
		printf("The number of equations nl should be set fistly\n");
		exit(0);
	}
	if ((nTotalCell == 0)||(nBoundFace == 0)){
		printf("Error: nTotalCell or nBoundFace should be read firstly\n");
		exit(0);
	}
	int nEquation = nl + nchem;
	dqdx = (fpkind **)malloc(nEquation * sizeof(fpkind*));
	dqdy = (fpkind **)malloc(nEquation * sizeof(fpkind*));
	dqdz = (fpkind **)malloc(nEquation * sizeof(fpkind*));
	int nTotal = nTotalCell + nBoundFace;
	size_t sizeQ = nTotal * nEquation * sizeof(fpkind);
	dqdx[0] = (fpkind *)malloc(sizeQ);
	dqdy[0] = (fpkind *)malloc(sizeQ);
	dqdz[0] = (fpkind *)malloc(sizeQ);
	int equationID;
	for (equationID = 1; equationID < nEquation; equationID++){
		dqdx[equationID] = &dqdx[equationID-1][nTotal];
		dqdy[equationID] = &dqdy[equationID-1][nTotal];
		dqdz[equationID] = &dqdz[equationID-1][nTotal];
	}
}

void setGradientQRandom(){
	if ((!dqdx)||(!dqdy)||(!dqdz)) {
		printf("Error: dqdx, dqdy, and dqdz should be allocated firstly\n");
		exit(0);
	}
	int nEquation = nl + nchem;
	int nTotal = nTotalCell + nBoundFace;
	int equationID, cellID;
	for (cellID = 0; cellID < nTotal; cellID++){
		for (equationID = 0; equationID < nEquation; equationID++){
			dqdx[equationID][cellID] = RANDOMNUMBER(-3.0, 4.0, nTotal);
			dqdy[equationID][cellID] = RANDOMNUMBER(-3.0, 4.0, nTotal);
			dqdz[equationID][cellID] = RANDOMNUMBER(-3.0, 4.0, nTotal);
		}
	}
}

void mallocLimit(){
	if (limit) return;
	if ((nTotalCell == 0)||(nBoundFace == 0)){
		printf("Error: nTotalCell or nBoundFace should be read firstly\n");
		exit(0);
	}
	int nTotal = nTotalCell + nBoundFace;
	limit = (fpkind *)malloc(sizeof(fpkind) * nTotal);
}

void setLimitRandom(){
	if (!limit) {
		printf("Error: limit should be allocated firstly\n");
		exit(0);
	}
	int nTotal = nTotalCell + nBoundFace;
	int cellID;
	for (cellID = 0; cellID < nTotal; cellID++){
		limit[cellID] = RANDOMNUMBER(3.0, 5.0, nTotal);
	}
}

void mallocRes(){
	if (res) return;
	if ((nTotalCell == 0)||(nBoundFace == 0)||( nl==0 )){
		printf("Error: nTotalCell, nBoundFace, and nl should be set firstly\n");
		exit(0);
	}
	int nTotal = nTotalCell + nBoundFace;
	int nEquation = nl + nchem;
	res = (fpkind **)malloc(sizeof(fpkind *) * nEquation);
	res[0] = (fpkind *)malloc(sizeof(fpkind) * nEquation * nTotal);
	resAVX = (fpkind **)malloc(sizeof(fpkind *) * nEquation);
	resAVX[0] = (fpkind *)malloc(sizeof(fpkind) * nEquation * nTotal);
	int equationID;
	for (equationID = 1; equationID < nEquation; equationID++){
		res[equationID] = &res[equationID-1][nTotal];
		resAVX[equationID] = &resAVX[equationID-1][nTotal];
	}
	//Initialize res
	int cellID;
	for (equationID = 0; equationID < nEquation; equationID++){
		for (cellID = 0; cellID < nTotal; cellID++){
			res[equationID][cellID] = 0.0;
			resAVX[equationID][cellID] = 0.0;
		}
	}
	
}

void setResRandom(){
	if (!res) {
		printf("Error: res should be allocated memory firstly\n");
		exit(0);
	}
	int equationID, cellID;
	int nTotal = nTotalCell + nBoundFace;
	int nEquation = nl + nchem;
	for (equationID = 0; equationID < nEquation; equationID++){
		for (cellID = 0; cellID < nTotal; cellID++){
			res[equationID][cellID] = fpkind(RANDOMNUMBER(-3.0, 5.0, nTotal));
			resAVX[equationID][cellID] = res[equationID][cellID];
		}
	}
}

void setResOrg(){
	if (resOrg) return;
	if (!res) {
		printf("Error: res should be allocated memory firstly\n");
		exit(0);
	}

	int nTotal = nTotalCell + nBoundFace;
	int nEquation = nl + nchem;
	resOrg = (fpkind **)malloc(sizeof(fpkind *) * nEquation);
	resOrg[0] = (fpkind *)malloc(sizeof(fpkind) * nEquation * nTotal);
	int equationID;
	for (equationID = 1; equationID < nEquation; equationID++){
		resOrg[equationID] = &resOrg[equationID-1][nTotal];
	}
	int cellID;
	for (equationID = 0; equationID < nEquation; equationID++){
		for (cellID = 0; cellID < nTotal; cellID++){
			resOrg[equationID][cellID] = res[equationID][cellID];
		}
	}
}

void mallocFlux(){
	if (flux) return;
	if ((SEG_LEN == 0)||(nl == 0)){
		printf("Error: nTotalFace and nl should be set firstly");
		exit(0);
	}
	int nEquation = nl + nchem;
	flux = (fpkind**)malloc(sizeof(fpkind *) * nEquation);
	flux[0] = (fpkind *)malloc(sizeof(fpkind) * nEquation * SEG_LEN);
	int equationID;
	for (equationID = 1; equationID < nEquation; equationID++){
		flux[equationID] = &flux[equationID-1][SEG_LEN];
	}
	//Initialize flux
	int faceID;
	for (equationID = 0; equationID < nEquation; equationID++){
		for (faceID = 0; faceID < SEG_LEN; faceID++){
			flux[equationID][faceID] = fpkind(RANDOMNUMBER(-2.0, 1.0, SEG_LEN));
		}
	}
}

void setFluxRandom(){
	if (!flux) {
		printf("Error: flux should be allocated memory firstly\n");
		exit(0);
	}
	int nEquation = nl + nchem;
	int faceID, equationID;
	for (equationID = 0; equationID < nEquation; equationID++){
		for (faceID = 0; faceID < SEG_LEN; faceID++){
			flux[equationID][faceID] = RANDOMNUMBER(-1.0, 2.0, SEG_LEN);
		}
	}
}

void mallocDMinDMax(){
	if ((dMin)&&(dMax)) return;
	if (!nTotalCell) {
		printf("Error: nTotalCell should be read from numberFaceOfEachCell\n");
		exit(0);
	}
	size_t sizeMinMax = nTotalCell * sizeof(fpkind);
	dMin = (fpkind *)malloc(sizeMinMax);
	dMax = (fpkind *)malloc(sizeMinMax);
	dMinAVX = (fpkind *)malloc(sizeMinMax);
	dMaxAVX = (fpkind *)malloc(sizeMinMax);
	int iCell;
	//initialize dMin and dMax by 0.0
	for (iCell = 0; iCell < nTotalCell; iCell++){
		dMin[iCell] = 0.0;
		dMax[iCell] = 0.0;
		dMinAVX[iCell] = 0.0;
		dMaxAVX[iCell] = 0.0;
	}
}

double RANDOMNUMBER(double upper, double lower, int numElement){
        if (numElement<=1){
                printf("Error: number of volumes is not larger than 1\n");
                exit(1);
        }
        //double x = ((double)rand()%numVolumes/(numVolumes-1));
        int x = rand()%numElement;
        return  lower + x*(upper-lower)/(numElement-1);
}

void freeGlobalVariablesHost(){
	DESTROYHOSTVAR(bcType);
	DESTROYHOSTVAR(leftCellofFace);
	DESTROYHOSTVAR(rightCellofFace);
	DESTROYHOSTVAR(cell2Face);
	DESTROYHOSTVAR(faceNumberOfEachCell);
	DESTROYHOSTVAR(xfn);
	DESTROYHOSTVAR(yfn);
	DESTROYHOSTVAR(zfn);
	DESTROYHOSTVAR(vgn);
	DESTROYHOSTVAR(xtn);
	DESTROYHOSTVAR(ytn);
	DESTROYHOSTVAR(ztn);
	DESTROYHOSTVAR(qNS);
	DESTROYHOSTVAR(qNSL);
	DESTROYHOSTVAR(qNSR);
	DESTROYHOSTVAR(face2Node);
	DESTROYHOSTVAR(nodeNumberOfEachFace);
	DESTROYHOSTVAR(face2NodePosition);
	DESTROYHOSTVAR(node2Cell);
	DESTROYHOSTVAR(cellNumberOfEachNode);
	DESTROYHOSTVAR(node2CellPosition);
	DESTROYHOSTVAR(cell2Node);
	DESTROYHOSTVAR(nodeNumberOfEachCell);
	DESTROYHOSTVAR(cell2NodePosition);
	DESTROYHOSTVAR(cell2NodeCount);
	DESTROYHOSTVAR(node2CellCount);
	DESTROYHOSTVAR(tCell);
	DESTROYHOSTVAR(qNode);
	DESTROYHOSTVAR(tNode);
	DESTROYHOSTVAR(nCount);
	DESTROYHOSTVAR(dMin);
	DESTROYHOSTVAR(dMax);
	DESTROYHOSTVAR(dMinAVX);
	DESTROYHOSTVAR(dMaxAVX);
}

void faceColor(){
	int * BoundFaceConflictPosi = (int *)malloc(nBoundFace * sizeof(int));
        int * BoundFaceConflictNum = (int *)malloc(nBoundFace * sizeof(int));
        int * InteriorFaceConflictPosi = (int *)malloc((nTotalFace - nBoundFace)*sizeof(int));
        int * InteriorFaceConflictNum = (int *)malloc((nTotalFace - nBoundFace)*sizeof(int));
	int sumBoundFaceConflict = 0;
	int sumInteriorFaceConflict = 0;
	int le, re;
//find face conflict relationship
	for (int i = 0; i < nBoundFace; i++){
		le = leftCellofFace[i];
		BoundFaceConflictPosi[i] = sumBoundFaceConflict;
		//add the number of face in left and right cell, the total number should except itself.
		sumBoundFaceConflict += faceNumberOfEachCell[le] - 1; 
	}

	for (int i = nBoundFace; i < nTotalFace; i++){
		le = leftCellofFace[i];
		re = rightCellofFace[i]; 
		InteriorFaceConflictPosi[i - nBoundFace] = sumInteriorFaceConflict;
		//add the number of face in left and right cell, the total number should except itself.
		sumInteriorFaceConflict += faceNumberOfEachCell[re] - 1; 
		sumInteriorFaceConflict += faceNumberOfEachCell[le] - 1; 
	}

	int * BoundFaceConflict = (int *)malloc(sumBoundFaceConflict*sizeof(int));
	int * InteriorFaceConflict = (int *)malloc(sumInteriorFaceConflict*sizeof(int));

	for (int i = 0; i < nBoundFace; i++){
		le = leftCellofFace[i];
		//Initializaion of BoundFaceConflictNum
		BoundFaceConflictNum[i] = 0;
		for (int j = 0; j < faceNumberOfEachCell[le]; j++) {
			if ( i != cell2Face[le][j]) {
				BoundFaceConflict[BoundFaceConflictPosi[i] + BoundFaceConflictNum[i]] = cell2Face[le][j];
				BoundFaceConflictNum[i]++;
			}
		}
	}


	for (int i = nBoundFace; i < nTotalFace; i++){
		le = leftCellofFace[i];
		re = rightCellofFace[i];
		int localFace = i - nBoundFace;
		//Initializaion of InteriorFaceConflictNum
		InteriorFaceConflictNum[localFace] = 0;
			//
		for (int j = 0; j < faceNumberOfEachCell[le]; j++) {
			if ( i != cell2Face[le][j]) {
				InteriorFaceConflict[InteriorFaceConflictPosi[localFace] + InteriorFaceConflictNum[localFace]] = cell2Face[le][j];
				InteriorFaceConflictNum[localFace]++;
			}
		}

		for (int j = 0; j < faceNumberOfEachCell[re]; j++) {
			if ( i != cell2Face[re][j]) {
				InteriorFaceConflict[InteriorFaceConflictPosi[localFace] + InteriorFaceConflictNum[localFace]] = cell2Face[re][j];
				InteriorFaceConflictNum[localFace]++;
			}
		}
	}

	//color conflict faces on boundary

	int colorMax = 0;
	BoundFaceGroup = (int *)malloc(nBoundFace*sizeof(int));
	InteriorFaceGroup = (int *)malloc((nTotalFace - nBoundFace)*sizeof(int));
	int * faceColor = (int *)malloc(nTotalFace*sizeof(int));
	for (int i = 0; i < nTotalFace; i++) {
		//Initialization of faceColor with 0, which means no color
		faceColor[i] = -1; 
	}

	for (int i = 0; i < nBoundFace; i++) {
		int color = 0;
		int colorSame = 0;
		while (faceColor[i] == -1) {
			for (int j = 0; j < BoundFaceConflictNum[i]; j++) {
				int faceConflict = BoundFaceConflict[BoundFaceConflictPosi[i]+j];
				if (color == faceColor[faceConflict]) {
					colorSame = 1;
					break;				
				}
			}
			if (colorSame == 0) faceColor[i] = color;
			else {
				color ++;
				colorSame = 0;
			}
		}
		//record the maximum color
		if (faceColor[i] > colorMax) colorMax = faceColor[i];
	}
	BoundFaceColorNum = colorMax + 1;
	printf("Boundary faces own %d colors\n", BoundFaceColorNum);
	BoundFaceGroupNum = (int *)malloc(BoundFaceColorNum*sizeof(int));
	BoundFaceGroupPosi =(int *)malloc(BoundFaceColorNum*sizeof(int));
	int * BoundFaceColorOffset = (int *)malloc(BoundFaceColorNum*sizeof(int));
	BoundFaceGroupPosi[0] = 0;
	for (int i = 0; i < BoundFaceColorNum; i++){
		//Initializaiton with zero
		BoundFaceGroupNum[i] = 0;
		BoundFaceColorOffset[i] = 0;
	}

	for (int i = 0; i < nBoundFace; i++) {
		int color = faceColor[i];
		BoundFaceGroupNum[color] ++;
	}

	for (int i = 1; i < BoundFaceColorNum; i++) {
		BoundFaceGroupPosi[i] = BoundFaceGroupPosi[i-1] + BoundFaceGroupNum[i-1];
	}

	for (int i = 0; i < BoundFaceColorNum; i++){
		//Initializaiton with zero
		BoundFaceColorOffset[i] = 0;
	}

	for (int i = 0; i < nBoundFace; i++) {
		int color = faceColor[i];
		int colorPosi = BoundFaceGroupPosi[color] + BoundFaceColorOffset[color];
		BoundFaceGroup[colorPosi] = i;
		BoundFaceColorOffset[color]++;
	}

	//color conflict faces in interior
	colorMax = 0; //reset colorMax for interior faces
	for (int i = 0; i < nTotalFace; i++) {
		faceColor[i] = -1; 
	}

	for (int i = nBoundFace; i < nTotalFace; i++) {
		int color = 0;
		int colorSame = 0;
		int localFace = i - nBoundFace;
		while (faceColor[i] == -1) {
			for (int j = 0; j < InteriorFaceConflictNum[localFace]; j++) {
				int faceConflict = InteriorFaceConflict[InteriorFaceConflictPosi[localFace]+j];
				if (color == faceColor[faceConflict]) {
					colorSame = 1;
					break;				
				}
			}
			if (colorSame == 0) faceColor[i] = color;
			else {
				color ++;
				colorSame = 0;
			}
		}
		//record the maximum color
		if (faceColor[i] > colorMax) colorMax = faceColor[i];
	}
	InteriorFaceColorNum = colorMax + 1;
	printf("The interior faces own %d colors\n", InteriorFaceColorNum);
	InteriorFaceGroupNum = (int*)malloc(InteriorFaceColorNum*sizeof(int));
	InteriorFaceGroupPosi = (int*)malloc(InteriorFaceColorNum*sizeof(int));
	int * InteriorFaceColorOffset = (int*)malloc(InteriorFaceColorNum*sizeof(int));
	InteriorFaceGroupPosi[0] = 0;

	for (int i = 0; i < InteriorFaceColorNum; i++){
		//Initializaiton with zero
		InteriorFaceGroupNum[i] = 0;
		InteriorFaceColorOffset[i] = 0;
	}

	for (int i = nBoundFace; i < nTotalFace; i++) {
		int color = faceColor[i];
		InteriorFaceGroupNum[color] ++;
	}

	for (int i = 1; i < InteriorFaceColorNum; i++) {
		InteriorFaceGroupPosi[i] = InteriorFaceGroupPosi[i-1] + InteriorFaceGroupNum[i-1];
	}
	for (int i = 0; i < InteriorFaceColorNum; i++){
		//Initializaiton with zero
		InteriorFaceColorOffset[i] = 0;
	}

	for (int i = nBoundFace; i < nTotalFace; i++) {
		int color = faceColor[i];
		int colorPosi = InteriorFaceGroupPosi[color] + InteriorFaceColorOffset[color];
		InteriorFaceGroup[colorPosi] = i;
		InteriorFaceColorOffset[color]++;
	}
}
void reorderFaceVars(){
	int equationID;
	leftCellofFaceRe = (int *)malloc(nTotalFace*sizeof(int));
	rightCellofFaceRe = (int *)malloc(nTotalFace*sizeof(int));
	int nEquation = nl + nchem;
	if (flux) {
		fluxRe = (fpkind**)malloc(sizeof(fpkind *) * nEquation);
		fluxRe[0] = (fpkind *)malloc(sizeof(fpkind) * nEquation * SEG_LEN);
		for (equationID = 1; equationID < nEquation; equationID++){
			fluxRe[equationID] = &fluxRe[equationID-1][SEG_LEN];
		}
	}

	for (int groupFaceID = 0; groupFaceID < nBoundFace; groupFaceID++){
		int faceID = BoundFaceGroup[groupFaceID];
		leftCellofFaceRe[groupFaceID] = leftCellofFace[faceID];
		rightCellofFaceRe[groupFaceID] = rightCellofFace[faceID];
		if (flux) {
			for (int equationID = 0; equationID < nEquation; equationID++){
				fluxRe[equationID][groupFaceID] = flux[equationID][faceID];
			}
		}
	}

	for (int groupFaceID = nBoundFace; groupFaceID < nTotalFace; groupFaceID++){
		int offset = groupFaceID - nBoundFace;
		int faceID = InteriorFaceGroup[offset];
		leftCellofFaceRe[groupFaceID] = leftCellofFace[faceID];
		rightCellofFaceRe[groupFaceID] = rightCellofFace[faceID];
		if (flux) {
			for (int equationID = 0; equationID < nEquation; equationID++){
				fluxRe[equationID][groupFaceID] = flux[equationID][faceID];
			}
		}
	}
}
void faceColorByNode(){
	printf("faceColorByNode\n");
	printf("nTotalFace = %d, nBoundFace = %d\n", nTotalFace, nBoundFace);
	int * BoundFaceConflictPosi = (int *)malloc(nBoundFace * sizeof(int));
        int * BoundFaceConflictNum = (int *)malloc(nBoundFace * sizeof(int));
        int * InteriorFaceConflictPosi = (int *)malloc((nTotalFace - nBoundFace)*sizeof(int));
        int * InteriorFaceConflictNum = (int *)malloc((nTotalFace - nBoundFace)*sizeof(int));
	int sumBoundFaceConflict = 0;
	int sumInteriorFaceConflict = 0;
	int le, re;
	vector<int> faceGroupTmp;
//find face conflict relationship
	printf("Boundary conflict\n");
	for (int i = 0; i < nBoundFace; i++){
		int numNodes = nodeNumberOfEachFace[i];	
		int nodePosi = face2NodePosition[i];
		for (int j = 0; j < numNodes; j++){
			int nodeID = face2Node[nodePosi + j];
			int numFaces = faceNumberOfEachNode[nodeID];
			int facePosi = node2FacePosition[nodeID];
			for (int k = 0; k < numFaces; k++){
				int faceID = node2Face[facePosi + k];	
				if (faceID != i) {
					vector<int>::iterator it = find(faceGroupTmp.begin(), faceGroupTmp.end(), faceID);
					if (it == faceGroupTmp.end()) faceGroupTmp.push_back(faceID);
				}
			}
		}
		BoundFaceConflictPosi[i] = sumBoundFaceConflict;
		sumBoundFaceConflict += faceGroupTmp.size();
		faceGroupTmp.clear();
		//add the number of face in left and right cell, the total number should except itself.
	}

	printf("Interior conflict, faceGroupTmp.size() = %d, sumBoundFaceConflict = %d\n", faceGroupTmp.size(), sumBoundFaceConflict);
	for (int i = nBoundFace; i < nTotalFace; i++){
		int numNodes = nodeNumberOfEachFace[i];	
		int nodePosi = face2NodePosition[i];
		for (int j = 0; j < numNodes; j++){
			int nodeID = face2Node[nodePosi + j];
			int numFaces = faceNumberOfEachNode[nodeID];
			int facePosi = node2FacePosition[nodeID];
			for (int k = 0; k < numFaces; k++){
				int faceID = node2Face[facePosi + k];	
				if (faceID != i) {
					vector<int>::iterator it = find(faceGroupTmp.begin(), faceGroupTmp.end(), faceID);
					if (it == faceGroupTmp.end()) faceGroupTmp.push_back(faceID);
				}
			}
			if (i == nBoundFace) printf("j = %d, numFaces = %d, nodeID = %d\n", j, numFaces, nodeID);
		}
		InteriorFaceConflictPosi[i - nBoundFace] = sumInteriorFaceConflict;
		//add the number of face in left and right cell, the total number should except itself.
		sumInteriorFaceConflict += faceGroupTmp.size(); 
		faceGroupTmp.clear();
	}

	int * BoundFaceConflict = (int *)malloc(sumBoundFaceConflict*sizeof(int));
	int * InteriorFaceConflict = (int *)malloc(sumInteriorFaceConflict*sizeof(int));

	printf("Boundary conflict, sumInteriorFaceConflict = %d\n", sumInteriorFaceConflict);
	for (int i = 0; i < nBoundFace; i++){
		int numNodes = nodeNumberOfEachFace[i];	
		int nodePosi = face2NodePosition[i];
		for (int j = 0; j < numNodes; j++){
			int nodeID = face2Node[nodePosi + j];
			int numFaces = faceNumberOfEachNode[nodeID];
			int facePosi = node2FacePosition[nodeID];
			for (int k = 0; k < numFaces; k++){
				int faceID = node2Face[facePosi + k];	
				if (faceID != i) {
					vector<int>::iterator it = find(faceGroupTmp.begin(), faceGroupTmp.end(), faceID);
					if (it == faceGroupTmp.end()) faceGroupTmp.push_back(faceID);
				}
			}
		}
		//Initializaion of BoundFaceConflictNum
		BoundFaceConflictNum[i] = faceGroupTmp.size();
		int facePosi = BoundFaceConflictPosi[i];
		for (int it = 0; it < faceGroupTmp.size(); it++){
			BoundFaceConflict[facePosi + it] = faceGroupTmp[it];
		}
		/*
		//for test
		if (i==0) {
			printf("facePosi = %d, faceGroupTmp.size() = %d\n", facePosi, faceGroupTmp.size());
			for (int it = 0; it < faceGroupTmp.size(); it++){
				printf("it = %d, BoundFaceConflict = %d, faceGroupTmp = %d\n", it, BoundFaceConflict[facePosi + it], faceGroupTmp[it]);
			}
		}
		*/
		faceGroupTmp.clear();
	}

	printf("Interior conflict\n");
	for (int i = nBoundFace; i < nTotalFace; i++){
		int numNodes = nodeNumberOfEachFace[i];	
		int nodePosi = face2NodePosition[i];
		for (int j = 0; j < numNodes; j++){
			int nodeID = face2Node[nodePosi + j];
			int numFaces = faceNumberOfEachNode[nodeID];
			int facePosi = node2FacePosition[nodeID];
			for (int k = 0; k < numFaces; k++){
				int faceID = node2Face[facePosi + k];	
				if (faceID != i) {
					vector<int>::iterator it = find(faceGroupTmp.begin(), faceGroupTmp.end(), faceID);
					if (it == faceGroupTmp.end()) faceGroupTmp.push_back(faceID);
				}
			}
		}
		//Initializaion of InteriorFaceConflictNum
		int localFace = i - nBoundFace;
		InteriorFaceConflictNum[localFace] = faceGroupTmp.size();
		int facePosi = InteriorFaceConflictPosi[localFace];
		for (int it = 0; it < faceGroupTmp.size(); it++){
			InteriorFaceConflict[facePosi + it] = faceGroupTmp[it];
		}
		faceGroupTmp.clear();
	}
	//color conflict faces on boundary
	int colorMax = 0;
	BoundFaceNodeGroup = (int *)malloc(nBoundFace*sizeof(int));
	InteriorFaceNodeGroup = (int *)malloc((nTotalFace - nBoundFace)*sizeof(int));
	int * faceColor = (int *)malloc(nTotalFace*sizeof(int));
	for (int i = 0; i < nTotalFace; i++) {
		//Initialization of faceColor with 0, which means no color
		faceColor[i] = -1; 
	}

	for (int i = 0; i < nBoundFace; i++) {
		int color = 0;
		int colorSame = 0;
		while (faceColor[i] == -1) {
			for (int j = 0; j < BoundFaceConflictNum[i]; j++) {
				int faceConflict = BoundFaceConflict[BoundFaceConflictPosi[i]+j];
				if (color == faceColor[faceConflict]) {
					colorSame = 1;
					break;				
				}
			}
			if (colorSame == 0) faceColor[i] = color;
			else {
				color ++;
				colorSame = 0;
			}
		}
		//record the maximum color
		if (faceColor[i] > colorMax) colorMax = faceColor[i];
	}
	BoundFaceNodeColorNum = colorMax + 1;
	printf("Boundary faces own %d colors due to nodes\n", BoundFaceNodeColorNum);
	BoundFaceNodeGroupNum = (int *)malloc(BoundFaceNodeColorNum*sizeof(int));
	BoundFaceNodeGroupPosi =(int *)malloc(BoundFaceNodeColorNum*sizeof(int));
	int * BoundFaceColorOffset = (int *)malloc(BoundFaceNodeColorNum*sizeof(int));
	BoundFaceNodeGroupPosi[0] = 0;
	for (int i = 0; i < BoundFaceNodeColorNum; i++){
		//Initializaiton with zero
		BoundFaceNodeGroupNum[i] = 0;
		BoundFaceColorOffset[i] = 0;
	}

	for (int i = 0; i < nBoundFace; i++) {
		int color = faceColor[i];
		BoundFaceNodeGroupNum[color] ++;
	}

	for (int i = 1; i < BoundFaceNodeColorNum; i++) {
		BoundFaceNodeGroupPosi[i] = BoundFaceNodeGroupPosi[i-1] + BoundFaceNodeGroupNum[i-1];
	}

	for (int i = 0; i < BoundFaceNodeColorNum; i++){
		//Initializaiton with zero
		BoundFaceColorOffset[i] = 0;
	}

	for (int i = 0; i < nBoundFace; i++) {
		int color = faceColor[i];
		int colorPosi = BoundFaceNodeGroupPosi[color] + BoundFaceColorOffset[color];
		BoundFaceNodeGroup[colorPosi] = i;
		BoundFaceColorOffset[color]++;
	}

	//color conflict faces in interior
	colorMax = 0; //reset colorMax for interior faces
	for (int i = 0; i < nTotalFace; i++) {
		faceColor[i] = -1; 
	}

	for (int i = nBoundFace; i < nTotalFace; i++) {
		int color = 0;
		int colorSame = 0;
		int localFace = i - nBoundFace;
		while (faceColor[i] == -1) {
			for (int j = 0; j < InteriorFaceConflictNum[localFace]; j++) {
				int faceConflict = InteriorFaceConflict[InteriorFaceConflictPosi[localFace]+j];
				if (color == faceColor[faceConflict]) {
					colorSame = 1;
					break;				
				}
			}
			if (colorSame == 0) faceColor[i] = color;
			else {
				color ++;
				colorSame = 0;
			}
		}
		//record the maximum color
		if (faceColor[i] > colorMax) colorMax = faceColor[i];
	}
	InteriorFaceNodeColorNum = colorMax + 1;
	printf("The interior faces own %d colors due to node\n", InteriorFaceNodeColorNum);
	InteriorFaceNodeGroupNum = (int*)malloc(InteriorFaceNodeColorNum*sizeof(int));
	InteriorFaceNodeGroupPosi = (int*)malloc(InteriorFaceNodeColorNum*sizeof(int));
	int * InteriorFaceColorOffset = (int*)malloc(InteriorFaceNodeColorNum*sizeof(int));
	InteriorFaceNodeGroupPosi[0] = 0;

	for (int i = 0; i < InteriorFaceNodeColorNum; i++){
		//Initializaiton with zero
		InteriorFaceNodeGroupNum[i] = 0;
		InteriorFaceColorOffset[i] = 0;
	}

	for (int i = nBoundFace; i < nTotalFace; i++) {
		int color = faceColor[i];
		InteriorFaceNodeGroupNum[color] ++;
	}

	for (int i = 1; i < InteriorFaceNodeColorNum; i++) {
		InteriorFaceNodeGroupPosi[i] = InteriorFaceNodeGroupPosi[i-1] + InteriorFaceNodeGroupNum[i-1];
	}
	for (int i = 0; i < InteriorFaceNodeColorNum; i++){
		//Initializaiton with zero
		InteriorFaceColorOffset[i] = 0;
	}

	for (int i = nBoundFace; i < nTotalFace; i++) {
		int color = faceColor[i];
		int colorPosi = InteriorFaceNodeGroupPosi[color] + InteriorFaceColorOffset[color];
		InteriorFaceNodeGroup[colorPosi] = i;
		InteriorFaceColorOffset[color]++;
	}
}

void reorderFaceVarsByNode(){
	int equationID;
	int nEquation = nl + nchem;
	leftCellofFaceNodeRe = (int *)malloc(nTotalFace*sizeof(int));
	rightCellofFaceNodeRe = (int *)malloc(nTotalFace*sizeof(int));
	nodeNumberOfEachFaceNodeRe = (int *)malloc(nTotalFace*sizeof(int)); //face coloring according to node
	face2NodePositionNodeRe = (int *)malloc(nTotalFace*sizeof(int)); //face coloring according to node
	int sumFace2Node = 0;
	for (int groupFaceID = 0; groupFaceID < nBoundFace; groupFaceID++){
		int faceID = BoundFaceNodeGroup[groupFaceID];
		if (faceID > nBoundFace) {
			printf("Error: groupFaceID = %d, faceID = %d\n", groupFaceID, faceID);
			exit(1);
		}
		leftCellofFaceNodeRe[groupFaceID] = leftCellofFace[faceID];
		rightCellofFaceNodeRe[groupFaceID] = rightCellofFace[faceID];
		nodeNumberOfEachFaceNodeRe[groupFaceID] = nodeNumberOfEachFace[faceID];
		face2NodePositionNodeRe[groupFaceID] = face2NodePosition[faceID];
		sumFace2Node += nodeNumberOfEachFace[faceID];
	}
	
	for (int groupFaceID = nBoundFace; groupFaceID < nTotalFace; groupFaceID++){
		int offset = groupFaceID - nBoundFace;
		int faceID = InteriorFaceNodeGroup[offset];
		if ((faceID < nBoundFace)||(faceID >= nTotalFace)) {
			printf("Error: groupFaceID = %d, faceID = %d\n", groupFaceID, faceID);
			exit(1);
		}
		leftCellofFaceNodeRe[groupFaceID] = leftCellofFace[faceID];
		rightCellofFaceNodeRe[groupFaceID] = rightCellofFace[faceID];
		nodeNumberOfEachFaceNodeRe[groupFaceID] = nodeNumberOfEachFace[faceID];
		face2NodePositionNodeRe[groupFaceID] = face2NodePosition[faceID];
		sumFace2Node += nodeNumberOfEachFace[faceID];
	}
	/*
	face2NodePositionNodeRe[0] = 0;
	for (int groupFaceID = 1; groupFaceID < nTotalFace; groupFaceID++){
		face2NodePositionNodeRe[groupFaceID] = face2NodePositionNodeRe[groupFaceID-1]+nodeNumberOfEachFaceNodeRe[groupFaceID-1];
	}
	*/
	/*
	int numNodes = 0;
	int iFace;
	for (iFace = 0; iFace < nTotalFace; iFace++){
		face2NodePositionNodeRe[iFace] = numNodes;
		numNodes += nodeNumberOfEachFaceNodeRe[iFace];
	}
	*/
	//face2Node = (int *)malloc(sizeof(int) * numNodes);
/*	
	//int * face2NodeNodeRe = (int *)malloc(numNodes * sizeof(int));
	//int * face2NodeNodeRe = (int *)malloc(sumFace2Node * sizeof(int));
	for (int groupFaceID = 0; groupFaceID < nBoundFace; groupFaceID++){
		int faceID = BoundFaceNodeGroup[groupFaceID];
		int nodePosiOrg = face2NodePosition[faceID];
		int nodePosiNew = face2NodePositionNodeRe[groupFaceID];
		
		int numNode = nodeNumberOfEachFaceNodeRe[groupFaceID];
		for (int offset = 0; offset < numNode; offset++){
			face2NodeNodeRe[nodePosiNew + offset] = face2Node[nodePosiOrg + offset];
		}
	}
	for (int groupFaceID = nBoundFace; groupFaceID < nTotalFace; groupFaceID++){
		int offset = groupFaceID - nBoundFace;
		int faceID = InteriorFaceNodeGroup[offset];
		int nodePosiOrg = face2NodePosition[faceID];
		int nodePosiNew = face2NodePositionNodeRe[groupFaceID];
		int numNode = nodeNumberOfEachFaceNodeRe[groupFaceID];
		for (int i = 0; i < numNode; i++){
			face2NodeNodeRe[nodePosiNew + i] = face2Node[nodePosiOrg + i];
		}
	}
	//check face2NodeNodeRe
	for (int faceID = 0; faceID < nTotalFace; faceID++){
		int numNode = nodeNumberOfEachFaceNodeRe[faceID];
		int nodePosi = face2NodePositionNodeRe[faceID];
		for (int nodeOffset = 0; nodeOffset < numNode; nodeOffset++){
			int nodeID = face2NodeNodeRe[nodePosi + nodeOffset];
			if ((nodeID < 0)||(nodeID >= nTotalNode)) {
				printf("Error: faceID = %d, numNode = %d, nodePosi = %d, nodeOffset = %d, nodeID = %d\n", faceID, numNode, nodePosi, nodeOffset, nodeID);
				exit(1);
			}
		}
	}
	printf("nodeNumberOfEachFaceNodeRe[%d] = %d, face2NodePositionNodeRe[%d] = %d, sumFace2Node = %d\n", nTotalFace-1, nodeNumberOfEachFaceNodeRe[nTotalFace-1], nTotalFace-1, face2NodePositionNodeRe[nTotalFace-1], sumFace2Node);
	for (int i = 0; i < sumFace2Node; i++){
		int nodeID = face2NodeNodeRe[i];
		//int nodeID = face2Node[i];
		if ((nodeID >= nTotalFace)||(nodeID < 0)){
			printf("Error, nTotalNode = %d, i = %d, nodeID = %d\n", nTotalNode, i, nodeID);
			exit(1);
		}
	}
*/

}

void cellColorByNode(){
	printf("cellColorByNode\n");
	printf("nTotalCell = %d\n", nTotalCell);
	int * CellConflictPosi = (int *)malloc(nTotalCell * sizeof(int));
        int * CellConflictNum = (int *)malloc(nTotalCell * sizeof(int));
	int sumCellConflict = 0;
	vector<int> cellGroupTmp;
//find cell conflict relationship
	for (int cellID = 0; cellID < nTotalCell; cellID++){
		int numNodes = nodeNumberOfEachCell[cellID];
		int nodePosi = cell2NodePosition[cellID];
		for (int nodeOffset = 0; nodeOffset < numNodes; nodeOffset++){
			int nodeID = cell2Node[nodePosi + nodeOffset];
			int numCells =  cellNumberOfEachNode[nodeID];
			int cellPosi = node2CellPosition[nodeID];
			for (int cellOffset = 0; cellOffset < numCells; cellOffset++){
				int cellNodeID = node2Cell[cellPosi + cellOffset];
				/*
				if (cellNodeID > nTotalCell) {
					printf("Error: cellID = %d, numNodes = %d, nodeID = %d, numCells = %d, cellNodeID = %d\n", cellID, numNodes, nodeID, numCells, cellNodeID);
					exit(1);
				}
				*/
				if ((cellNodeID != cellID)&&(cellNodeID < nTotalCell)) {
					vector<int>::iterator it = find(cellGroupTmp.begin(), cellGroupTmp.end(), cellNodeID);
					if (it == cellGroupTmp.end()) cellGroupTmp.push_back(cellNodeID);
				}
				
			}
		}
		CellConflictPosi[cellID] = sumCellConflict;
		sumCellConflict += cellGroupTmp.size();
		CellConflictNum[cellID] = cellGroupTmp.size();
		cellGroupTmp.clear();

	}

	int * CellConflict = (int *)malloc(sumCellConflict * sizeof(int));

	for (int cellID = 0; cellID < nTotalCell; cellID++){
		int numNodes = nodeNumberOfEachCell[cellID];
		int nodePosi = cell2NodePosition[cellID];
		for (int nodeOffset = 0; nodeOffset < numNodes; nodeOffset++){
			int nodeID = cell2Node[nodePosi + nodeOffset];
			int numCells =  cellNumberOfEachNode[nodeID];
			int cellPosi = node2CellPosition[nodeID];
			for (int cellOffset = 0; cellOffset < numCells; cellOffset++){
				int cellNodeID = node2Cell[cellPosi + cellOffset];
				if ((cellNodeID != cellID)&&(cellNodeID < nTotalCell)) {
					vector<int>::iterator it = find(cellGroupTmp.begin(), cellGroupTmp.end(), cellNodeID);
					if (it == cellGroupTmp.end()) cellGroupTmp.push_back(cellNodeID);
				}
				
			}
		}

		int cellPosi = CellConflictPosi[cellID];
		for (int it = 0; it < cellGroupTmp.size(); it++){
			CellConflict[cellPosi + it] = cellGroupTmp[it];
		}
		cellGroupTmp.clear();

	}
	//check
	for (int i = 0; i < sumCellConflict; i++){
		int cellID = CellConflict[i];
		if (cellID >= nTotalCell) {
			printf("i = %d, cellID = %d\n", i ,cellID);
			exit(1);
		}
	}
	//color conflict faces on boundary
	int colorMax = 0;
	CellNodeGroup = (int *)malloc(nTotalCell*sizeof(int));
	int * cellColor = (int *)malloc(nTotalCell*sizeof(int));
	for (int i = 0; i < nTotalCell; i++) {
		//Initialization of faceColor with 0, which means no color
		cellColor[i] = -1; 
	}
	for (int i = 0; i < nTotalCell; i++) {
		int color = 0;
		int colorSame = 0;
		/*
		printf("i = %d, nTotalCell = %d, CellConflictNum = %d, CellConflictPosi = %d, sumCellConflict = %d\n", i, nTotalCell, CellConflictNum[i], CellConflictPosi[i], sumCellConflict);
		for (int j = 0; j < CellConflictNum[i]; j++) {
			int cellConflictID = CellConflict[CellConflictPosi[i]+j];
			printf("%d, ", cellConflictID);
		}
		printf("\n");
		*/
		
		while (cellColor[i] == -1) {
			for (int j = 0; j < CellConflictNum[i]; j++) {
				int cellConflictID = CellConflict[CellConflictPosi[i]+j];
				if (color == cellColor[cellConflictID]) {
					colorSame = 1;
					break;				
				}
			}
			if (colorSame == 0) cellColor[i] = color;
			else {
				color ++;
				colorSame = 0;
			}
		}
		//record the maximum color
		if (cellColor[i] > colorMax) colorMax = cellColor[i];
		//printf("i = %d, cellColor = %d\n", i, cellColor[i]);
	}
	CellNodeColorNum = colorMax + 1;
	printf("Cells  own %d colors due to nodes\n", CellNodeColorNum);
	
	CellNodeGroupNum = (int *)malloc(CellNodeColorNum*sizeof(int));
	CellNodeGroupPosi =(int *)malloc(CellNodeColorNum*sizeof(int));
	int * CellColorOffset = (int *)malloc(CellNodeColorNum*sizeof(int));
	CellNodeGroupPosi[0] = 0;
	for (int i = 0; i < CellNodeColorNum; i++){
		//Initializaiton with zero
		CellNodeGroupNum[i] = 0;
		CellColorOffset[i] = 0;
	}

	for (int i = 0; i < nTotalCell; i++) {
		int color = cellColor[i];
		CellNodeGroupNum[color] ++;
	}

	for (int i = 1; i < CellNodeColorNum; i++) {
		CellNodeGroupPosi[i] = CellNodeGroupPosi[i-1] + CellNodeGroupNum[i-1];
	}

	for (int i = 0; i < CellNodeColorNum; i++){
		//Initializaiton with zero
		CellColorOffset[i] = 0;
	}

	for (int i = 0; i < nTotalCell; i++) {
		int color = cellColor[i];
		int colorPosi = CellNodeGroupPosi[color] + CellColorOffset[color];
		CellNodeGroup[colorPosi] = i;
		CellColorOffset[color]++;
	}
	//check cell coloring results
	int * colorCellNum = (int *)malloc(nTotalCell * sizeof(int));
	for (int cellID = 0; cellID < nTotalCell; cellID++){
		colorCellNum[cellID] = 0;
	}
	for (int colorID = 0; colorID < CellNodeColorNum; colorID++){
		int colorPosi = CellNodeGroupPosi[colorID];
		int groupNum = CellNodeGroupNum[colorID];
		for (int offset = 0; offset < groupNum; offset++){
			int cellID = CellNodeGroup[colorPosi + offset];
			colorCellNum[cellID] += 1;
			if ((cellID < 0)||(cellID >= nTotalCell)){
				printf("Error: colorID = %d, offset = %d, cellID = %d\n", colorID, offset, cellID);
				exit(1);
			}
		}
	}
	
	for (int cellID = 0; cellID < nTotalCell; cellID++){
		if (colorCellNum[cellID] != 1) {
			printf("Error, cellID = %d, colorCellNum = %d\n", cellID, colorCellNum[cellID]);
			exit(0);
		}
	}

	free(CellConflict);
	free(CellConflictPosi);
	free(CellConflictNum);
	free(colorCellNum);


}
void reorderCellVarsByNode(){
	nodeNumberOfEachCellRe = (int *)malloc(nTotalCell * sizeof(int));
	cell2NodePositionRe = (int *)malloc(nTotalCell * sizeof(int));
	int nEquation = nl + nchem;
	int nTotal = nTotalCell + nBoundFace;
	//allocate memory for qNS
	qNSRe = (fpkind **)malloc(sizeof(fpkind*)*nl);
	qNSRe[0] = (fpkind *)malloc(sizeof(fpkind)*nl*nTotalCell);
	for (int equationID = 1; equationID < nl; equationID++){
		qNSRe[equationID] = &qNSRe[equationID-1][nTotalCell];
	}
	//reorder variables by cell coloring
	for (int colorCellID = 0; colorCellID < nTotalCell; colorCellID++){
		int cellIDOrg = CellNodeGroup[colorCellID];
		nodeNumberOfEachCellRe[colorCellID] = nodeNumberOfEachCell[cellIDOrg];
		cell2NodePositionRe[colorCellID] = cell2NodePosition[cellIDOrg];
		for (int equationID = 0; equationID < nEquation; equationID++){
			qNSRe[equationID][colorCellID] = qNS[equationID][cellIDOrg];
		}
	}		

	int numNodes = 0;
	for (int cellID = 0; cellID < nTotalCell; cellID ++){
		numNodes += nodeNumberOfEachCell[cellID];
	}
	size_t sizeNodes = numNodes * sizeof(fpkind);
	cell2NodeCountRe = (fpkind *)malloc(sizeNodes);
	for (int countID = 0; countID < numNodes; countID++){
		cell2NodeCountRe[countID] = (fpkind)cell2NodeCount[countID];
	}
	
}
