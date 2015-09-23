#!/bin/bash

scriptName=`basename $0`
exName=${scriptName:0:${#scriptName}-3}

cascSuffix="-cascades.txt"
netSuffix="-network.txt"

cascName="data/""$exName"

minAlpha="0.0"
maxAlpha="0.2"
nodeNm="64"
edgeNm="128"
ROCPointNm="100"

./bin/generate_DCnets -g:"0.987 0.571;0.571 0.049" -ar:"0.05;""$maxAlpha" -c:1000 -f:"$cascName" -n:"$nodeNm" -e:"$edgeNm" -K:3 -m:0 

mergeCasc="$cascName""$cascSuffix"
mergeNet="$cascName""$netSuffix"
mergeOut="data/""$exName"

./bin/DataMerger -i:"$mergeCasc" -n:"$mergeNet" -o:"$mergeOut" -la:"$minAlpha" -ua:"$maxAlpha"

InfoPathOut="result/""$exName""-InfoPath"
MixCascadesOut="result/""$exName""-MixCascades"
MMRateOut="result/""$exName""-MMRate"
DecayCascadesOut="result/""$exName""-DecayCascades"

./bin/InfoPath -i:"$mergeOut""$cascSuffix" -n:"$mergeOut""$netSuffix" -o:"$InfoPathOut" -rm:3 -la:"$minAlpha" -ua:"$maxAlpha" -s:0 -e:1000 -g:0.005 -bl:10 -w:5 -m:0
./bin/MixCascades -i:"$mergeOut""$cascSuffix" -n:"$mergeOut""$netSuffix" -o:"$MixCascadesOut" -rm:3 -la:"$minAlpha" -ua:"$maxAlpha" -s:0 -em:10 -K:3 -e:100 -bl:10 -g:0.005 -w:5 -m:0
./bin/MMRate -i:"$mergeOut""$cascSuffix" -n:"$mergeOut""$netSuffix" -o:"$MMRateOut" -rm:3 -la:"$minAlpha" -ua:"$maxAlpha" -s:0 -em:10 -K:3 -e:100 -bl:10 -g:0.005 -w:5 -m:0
./bin/DecayCascades -i:"$mergeOut""$cascSuffix" -n:"$mergeOut""$netSuffix" -o:"$DecayCascadesOut" -rm:3 -la:"$minAlpha" -ua:"$maxAlpha" -s:0 -K:3 -em:10 -e:100 -bl:10 -g:0.005 -w:5 -m:0 

txtSuffix=".txt"
MaxTxtSuffix="_Max.txt"
plotOut="plot/""$exName"
modelNames="InfoPath":"MixCascades":"MMRate":"DecayCascades"
./bin/EvaluationAUC -i:"$InfoPathOut""$txtSuffix":"$MixCascadesOut""$txtSuffix":"$MMRateOut""$txtSuffix":"$DecayCascadesOut""$txtSuffix" -n:"$mergeOut""$netSuffix" -o:"$plotOut" -m:"$modelNames" -ua:"$maxAlpha" -p:"$ROCPointNm"
./bin/EvaluationMSE -i:"$InfoPathOut""$txtSuffix":"$MixCascadesOut""$MaxTxtSuffix":"$MMRateOut""$MaxTxtSuffix":"$DecayCascadesOut""$MaxTxtSuffix" -n:"$mergeOut""$netSuffix" -o:"$plotOut" -m:"$modelNames" -ua:"$maxAlpha" -p:"$ROCPointNm"
./bin/EvaluationMultiple -i:"$MixCascadesOut":"$MMRateOut":"$DecayCascadesOut" -n:"$mergeOut""$netSuffix" -o:"$plotOut" -m:"MixCascades":"MMRate":"DecayCascades" -ua:"$maxAlpha" -p:"$ROCPointNm"

