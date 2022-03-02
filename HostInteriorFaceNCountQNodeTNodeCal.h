//host function of InteriorFaceNCountQNodeTNodeCal
void CallHostInteriorFaceNCountQNodeTNodeCal(const int loopID);
//host function of boundary + interior face loop NcountQNodeTNode calculate
void CallHostFaceNCountQNodeTNodeCal(const int loopID);
//host function of boundary + interior node loop NcountQNodeTNode calculate
void CallHostNodeLoopNCountQNodeTNodeCal(const int loopID);
void CallHostNodeLoopNCountQNodeTNodeCalModify(const int loopID);
void CallHostNodeLoopNCountQNodeTNodeCalFinal(const int loopID);
//host function of boundary + interior cell loop NcountQNodeTNode calculate
void CallHostCellLoopNCountQNodeTNodeCal(const int loopID);
void CallHostCellLoopNCountQNodeTNodeCalFinal(const int loopID);
//initialize qNode and tNode as zero on both host and device
void initNCountQNodeTNodeZero(const int loopID);
void initNCountQNodeTNodeAVXZero(const int loopID);
//average qNode and tNode by nCount
void HostAverageQNodeTNode(const int loopID);
