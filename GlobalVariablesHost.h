#include "precision.hxx"
#include "commonVars.h"
//#include "constantVars.hxx"
#define DESTROYHOSTVAR(hostVar) if (hostVar) free(hostVar)
//Global Variables for CPU
extern int * bcType; //boundary type array
extern int * leftCellofFace; //Topology information, face's owner cell
extern int * rightCellofFace;//Topology information face's neighbour cell
extern int * leftCellofFaceRe; //reorder leftCellofFace by face coloring
extern int * rightCellofFaceRe;//reorder
extern int ** cell2Face; //Topology information, cell's faces
extern int ** cell2Cell; //Topology information, cell's cell, they share the same face
extern int ** cell2FaceRe; //reorder cell2Face's face order, put left faces together and ahead
extern int ** cell2CellRe; //reorder cell2Cell's cell order, similar to cell2FaceRe
extern int * faceNumberOfEachCell; //Topology information, face number owned by a cell
extern int * leftFaceNumberOfEachCell; //Topology information, left face number owned by cells
extern int * cell2FacePosition; //Topology information, start position of a cell in cell2Face
extern int ** leftRightFace; //Topology information, determine a face is a left or right face for a cell
extern int * face2Node; //Topology information, face's nodes, use with face2NodePosition
extern int * nodeNumberOfEachFace; //Topology information, node number owned by a face
extern int * face2NodePosition; //The start position of one face in face2Node
extern int * node2Cell; //Topology information, node's cells, use with node2CellPosition
extern int * cellNumberOfEachNode; //Topology information, cell number owned by a node
extern int * node2CellPosition; //The start position of one node in node2Cell
extern int * node2CellCount; //The cell-access frequency by a node. Or how many nodes are owned by a cell.
extern int * cell2Node; //Topology information, cell's nodes
extern int * cell2NodeCount; //The cell-access frequency by a node, sorted by cell.
extern int * nodeNumberOfEachCell; //Topology information, node number owned by a cell
extern int * cell2NodePosition; //The start position of one cell in cell2Node
extern int * node2Face; //Topology information, node's faces
extern int * faceNumberOfEachNode; //Topology information, face number owned by a node
extern int * node2FacePosition; //Topology information, start position of face by a node
extern int * BoundFaceGroup; //The order of face label in different colors
extern int * BoundFaceGroupPosi; //The offset of the first face label of one face group in BoundFaceGroup
extern int * BoundFaceGroupNum; //the number of every face group (the number of faces in one color)
extern int BoundFaceColorNum; //the total number of color (the total number of face groups)
extern int * InteriorFaceGroup;
extern int * InteriorFaceGroupPosi;
extern int * InteriorFaceGroupNum;
extern int InteriorFaceColorNum;
extern int * isGhostCell; //Topology information, determine whether or not is a ghost cell.
extern double * xfn; //Geometry variable, x component of face normal
extern double * yfn;
extern double * zfn;
extern double * vgn; //Velocity variable, normal velocity of face center
extern double * xtn; //Velocity variable, tangential velocity of face center
extern double * ytn;
extern double * ztn;
extern double * area; //area of faces
extern double * xfc;  //face center x component
extern double * yfc;
extern double * zfc;
extern double * xcc; //cell center y component
extern double * ycc;
extern double * zcc;
extern fpkind ** qNS; //rho, u, v, w, p
extern fpkind ** qNSL; //qNS on left cell of a face
extern fpkind ** qNSR; //qNS on right cell of a face
extern fpkind ** qNSLOrg; //get qNSL for a new host function
extern fpkind ** qNSROrg;
extern fpkind ** qNSLRFaceCpy; //qNSL and qNSR together
extern fpkind ** dqdx;
extern fpkind ** dqdy;
extern fpkind ** dqdz;
extern fpkind ** tCell; //temperature on cells
extern fpkind ** res; //residual for ns equations
extern fpkind ** resOrg; //store the original res for repeat computing
extern fpkind ** resAVX; //residual for ns equations by AVX512
extern fpkind ** flux; //flux for 5 unkonwns rho, u, v, w, p on faces.
extern fpkind ** fluxRe; //reorder flux by face coloring
extern fpkind ** qNode; //qNS on nodes 
extern fpkind * limit; //limit on faces 
extern fpkind ** tNode; // temperature on nodes
extern int * nCount; //cell number of each node by calculation. what's the difference between nCount and cellNumberOfEachNode (nCPN)??
extern fpkind * dMin; //local minimum value of q on cells, by calculation
extern fpkind * dMax; //local maxmum value of q on cells, by calculation
extern fpkind * dMinAVX; //AVX
extern fpkind * dMaxAVX; //AVX
void setPropertyOfSimulation();
void setSEGLEN();
void readLeftRightCellofFace();
void readBoundaryType();
void readNumberFaceOfEaceCell();
void readCell2Face();
void readNumberNodeOfEachFace();
void readFace2Node();
void readNumberCellOfEachNode();
void readNode2Cell();
void readNumberNodeOfEachCell();
void readCell2Node();
void setNode2CellCount();
void setCell2NodeCount();
void setLeftRightFace();
void setFaceNumberOfEachNode();
void setNode2Face();
void setIsGhostCellHost();
void setCell2Cell(); //set cell2Cell by cell2Face
void setCell2FaceRe(); //reorder cell2Face
void setCell2CellRe(); //reorder cell2Cell
void setXYZfnRandom();
double RANDOMNUMBER(double upper, double lower, int numElement);
void setVgnRandom();
void setXYZtnRandom();
void setAreaRandom();
void setXYZfcRandom();
void setXYZccRandom();
void setQNSRandom();
void mallocGradientQ();
void setGradientQRandom();
void setTCellRandom();
void mallocQNSLR();
void mallocCpyQNSLROrg();
void mallocQNSLRFaceCpy();
void setQNSLRRandom();
void mallocQNode();
void setQNodeRandom();
void mallocTNode();
void mallocNCount();
void mallocLimit();
void setLimitRandom();
void mallocRes();
void setResRandom();
void setResOrg();
void mallocFlux();
void setFluxRandom();
void mallocDMinDMax();
void faceColor();
void reorderFaceVars();
void freeGlobalVariablesHost();
