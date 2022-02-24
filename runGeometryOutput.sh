#!/bin/bash
rootPath="/GPUFS/sysu_xizhang2_1/PHengLEITestCases/testCasesM6"
testPath="/GPUFS/sysu_xizhang2_1/OptAVX512/test"
folder="100 200 400common 600 900"
#folder="900"
#outputPath=$testPath"/NoReOrder"
outputPath=$testPath"/ReOrder"
#judge whether or not outputPath exist
if [ -d $outputPath ]
then
	rm -rf $outputPath"/*"
else
	mkdir $outputPath
fi
#run case and copy output path to 
for element in $folder
do
	cd $rootPath"/"$element
	mpirun -n 28 ../../PHengLEIv3d0CPUGeometryOutput
	dataPath=$element"w_28"
	if [ -d $dataPath ]
	then
		rm -rf $dataPath"/*"
	else
		mkdir $dataPath
	fi
	mv boundaryType face2Node numberCellOfEachNode numberNodeOfEachFace cell2Face leftCellOfFace numberFaceOfEachCell rightCellOfFace cell2Node node2Cell numberNodeOfEachCell $dataPath
	cp -rf $dataPath $outputPath
done
