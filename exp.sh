#!/bin/bash

scriptName=`basename $0`
expName=${scriptName:0:${#scriptName}-3}

cascSuffix="-cascades.txt"
netSuffix="-network.txt"

cascName="data/""$expName"

minAlpha="0.0"
maxAlpha="0.2"
nodeNm="64"
edgeNm="128"

./bin/generate_DCnets -g:"0.987 0.571;0.571 0.049" -ar:"0.05;""$maxAlpha" -c:1000 -f:"$cascName" -n:"$nodeNm" -e:"$edgeNm" -K:3 -m:0 

InfoPathOut="result/""$expName""-InfoPath"
MixCascadesOut="result/""$expName""-MixCascades"
MMRateOut="result/""$expName""-MMRate"
DecayCascadesOut="result/""$expName""-DecayCascades"

./bin/InfoPath -i:"$cascName""$cascSuffix" -n:"$cascName""$netSuffix" -o:"$InfoPathOut" -rm:3 -la:"$minAlpha" -ua:"$maxAlpha" -s:0 -e:1000 -g:0.005 -bl:10 -w:5 -m:0
./bin/MixCascades -i:"$cascName""$cascSuffix" -n:"$cascName""$netSuffix" -o:"$MixCascadesOut" -rm:3 -la:"$minAlpha" -ua:"$maxAlpha" -s:0 -em:10 -K:3 -e:100 -bl:10 -g:0.005 -w:5 -m:0
./bin/MMRate -i:"$cascName""$cascSuffix" -n:"$cascName""$netSuffix" -o:"$MMRateOut" -rm:3 -la:"$minAlpha" -ua:"$maxAlpha" -s:0 -em:10 -K:3 -e:100 -bl:10 -g:0.005 -w:5 -m:0
./bin/DecayCascades -i:"$cascName""$cascSuffix" -n:"$cascName""$netSuffix" -o:"$DecayCascadesOut" -rm:3 -la:"$minAlpha" -ua:"$maxAlpha" -s:0 -K:3 -em:10 -e:100 -bl:10 -g:0.005 -w:5 -m:0 

txtSuffix=".txt"
MaxTxtSuffix="_Max.txt"
plotOut="plot/""$expName"
modelNames="InfoPath":"MixCascades":"MMRate":"DecayCascades"
./bin/EvaluationAUC -i:"$InfoPathOut""$txtSuffix":"$MixCascadesOut""$txtSuffix":"$MMRateOut""$txtSuffix":"$DecayCascadesOut""$txtSuffix" -n:"$cascName""$netSuffix" -o:"$plotOut" -m:"$modelNames" -ua:"$maxAlpha" 
./bin/EvaluationMSE -i:"$InfoPathOut""$txtSuffix":"$MixCascadesOut""$MaxTxtSuffix":"$MMRateOut""$MaxTxtSuffix":"$DecayCascadesOut""$MaxTxtSuffix" -n:"$cascName""$netSuffix" -o:"$plotOut" -m:"$modelNames" -ua:"$maxAlpha" 
./bin/EvaluationMultiple -i:"$MixCascadesOut":"$MMRateOut":"$DecayCascadesOut" -n:"$cascName""$netSuffix" -o:"$plotOut" -m:"MixCascades":"MMRate":"DecayCascades" -ua:"$maxAlpha" 

