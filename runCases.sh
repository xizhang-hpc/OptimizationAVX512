#!/bin/bash
rootPath="/GPUFS/sysu_xizhang2_1/OptAVX512/test"
casePath="ReOrder"
caseFolders="100w_28 200w_28 400commonw_28 600w_28 900w_28"
exeNames=(runT5000 runT2500 runT1000 runT700 runT400)
index=0
for element in $caseFolders
do
	testPath=$rootPath"/"$casePath"/"$element
	executeName=${exeNames[$index]}
	outputFile=$testPath"/"$outputFileName
	echo $testPath
	echo $executeName
	echo $outputFile
	cd $testPath
	$rootPath"/"$executeName
	let index++
done
