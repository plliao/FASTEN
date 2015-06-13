#include <Evaluator.h>
#include <InfoPathFileIO.h>

void Evaluator::LoadGroundTruth(TSIn &SIn) {
   NodeInfo nodeInfo; 
   InfoPathFileIO::LoadNetworkTxt(SIn, GroundTruth, nodeInfo);
}

void Evaluator::LoadInferredNetwork(TSIn &SIn, TStr &modelName) {
   NodeInfo nodeInfo;
   InferredNetworks.Add(TStrFltFltHNEDNet());
   ModelNames.Add(modelName);
   TStrFltFltHNEDNet &inferredNetwork = InferredNetworks.Last();
   InfoPathFileIO::LoadNetworkTxt(SIn, inferredNetwork, nodeInfo);
}

void Evaluator::EvaluatePRC(TFlt minAlpha, TFlt maxAlpha, const TFlt &step, TFlt PRCPointNm) {
   TFlt groundTruthTimeStep = GetGroundTruthTimeStep(step); 

   for (int i=0;i<InferredNetworks.Len();i++) {
      TFlt inferredStep = GetInferredTimeStep(step, i);
      PRC.Add(DyPRCPoints());
      DyPRCPoints &dyPRCPoints = PRC[i];
      if (dyPRCPoints.IsKey(step)) continue;
     
      dyPRCPoints.AddDat(step,PRCPoints());
      PRCPoints &prcPoints = dyPRCPoints.GetDat(step);

      printf("Evluating PRC points, model:%s\n",ModelNames[i]());

      TStrFltFltHNEDNet &inferredNetwork = InferredNetworks[i];
      TFlt nodeSize = (TFlt)GroundTruth.GetNodes();
      TFlt P = (TFlt)GroundTruth.GetEdges(), N = nodeSize * (nodeSize - 1.0) - P;
      TFlt PP = (TFlt)inferredNetwork.GetEdges();
      TFlt TP = 0.0, FP = 0.0, FN = 0.0;
      THash<TIntPr,TFlt> edgesAlphaVector;
      THash<TIntPr,bool> edgesTruthTable;
      prcPoints.Reserve((int)P);

      PRCPoint endPoint; 
      endPoint.Val1 = 1.0; endPoint.Val2 = P / (P + N);
      prcPoints.Add(endPoint);

      for (TStrFltFltHNEDNet::TEdgeI EI = inferredNetwork.BegEI(); EI < inferredNetwork.EndEI(); EI++) {
         TInt srcNId = EI.GetSrcNId(), dstNId = EI.GetDstNId();
         TIntPr index; index.Val1 = srcNId; index.Val2 = dstNId;
         
         edgesAlphaVector.AddDat(index, EI.GetDat().GetDat(inferredStep));
         if (GroundTruth.IsEdge(srcNId,dstNId)) { 
            edgesTruthTable.AddDat(index,true);
            TP++;
         }
         else {
            edgesTruthTable.AddDat(index,false);
            FP++;
         }
      }
      FN = P - TP;

      edgesAlphaVector.SortByDat();
         
      if (P!=TP) {
         TFlt ratio = (N-FP) / (P-TP);
         for (TFlt x = P - TP - 1; x >= 0.0; x--) {
            PRCPoint prcPoint;
            prcPoint.Val1 = (TP+x) / (P);
            prcPoint.Val2 = (TP+x) / (TP + x + FP + ratio * x);
            prcPoints.Add(prcPoint);
         }
      }
      else {
         PRCPoint prcPoint;
         prcPoint.Val1 = 1.0;
         prcPoint.Val2 = TP / (TP + FP);
         prcPoints.Add(prcPoint);
      }

      for (THash<TIntPr,TFlt>::TIter AI = edgesAlphaVector.BegI(); !AI.IsEnd(); AI++) {         
         TIntPr index = AI.GetKey();
         PRCPoint prcPoint;
         if (edgesTruthTable.GetDat(index)) {
            TP--;
            FN++;
         }
         else {
            FP--;
         }
         if (TP==0.0) break;
         prcPoint.Val1 = TP / P;
         prcPoint.Val2 = TP / (TP + FP);
         printf("Threshold: %f, Precision: %f, Recall: %f, x,y: %f,%f\n", AI.GetDat()(), prcPoint.Val2(), prcPoint.Val1(), prcPoint.Val1(), prcPoint.Val2());
         prcPoints.Add(prcPoint);
      }
      
      endPoint.Val1 = 0.0;
      endPoint.Val2 = 1.0;
      prcPoints.Add(endPoint);
   }
}

void Evaluator::EvaluateAUC(const TFlt &step) {

   if (PRC.Len()==0) return;
   for (int i=0;i<PRC.Len();i++) {
      TFlt sum = 0.0;
      DyPRCPoints &dyPRCPoints = PRC[i];
      PRCPoints &prcPoints = dyPRCPoints.GetDat(step); 
      int prcPointNm = prcPoints.Len();
      for (int j=0;j<prcPointNm-1;j++) {
         TFlt scale = TFlt::Abs(prcPoints[j].Val1 - prcPoints[j+1].Val1);
         TFlt hight = TFlt::GetMn(prcPoints[j].Val2,prcPoints[j+1].Val2);
         TFlt delta = TFlt::Abs(prcPoints[j].Val2 - prcPoints[j+1].Val2);
         sum += scale * ( hight + delta / 2);
      }
      printf("%s AUC: %f\n", ModelNames[i].CStr(), sum());
   }
}

void Evaluator::EvaluateMSE(const TFlt &step) {

  TFlt groundTruthStep = GetGroundTruthTimeStep(step);

  for (int i=0;i<InferredNetworks.Len();i++) { 
     MSE.Add(TFltFltH()); MAE.Add(TFltFltH());
     TFlt mse = 0.0;
     TFlt mae = 0.0;
     TStrFltFltHNEDNet &inferredNetwork = InferredNetworks[i];
     TFlt inferredStep = GetInferredTimeStep(step, i);

     for (TStrFltFltHNEDNet::TEdgeI EI = inferredNetwork.BegEI();EI!=inferredNetwork.EndEI();EI++) {
        int srcNId = EI.GetSrcNId();
        int dstNId = EI.GetDstNId();
        if (srcNId==dstNId) continue;
        
        TFlt inferredAlpha = EI.GetDat().GetDat(inferredStep);
        
        if (GroundTruth.IsEdge(srcNId,dstNId)) {
           TFlt groundTruthAlpha = GroundTruth.GetEDat(srcNId,dstNId).GetDat(groundTruthStep);
           mse += TMath::Sqr(inferredAlpha - groundTruthAlpha);
           mae += TFlt::Abs(inferredAlpha - groundTruthAlpha);
        }
        else {
           mse += TMath::Sqr(inferredAlpha);   
           mae += TFlt::Abs(inferredAlpha);
        }
     }

     for (TStrFltFltHNEDNet::TEdgeI EI = GroundTruth.BegEI();EI!=GroundTruth.EndEI();EI++) {
        int srcNId = EI.GetSrcNId();
        int dstNId = EI.GetDstNId();
        if (srcNId==dstNId) continue;

        if (!inferredNetwork.IsEdge(srcNId,dstNId)) {
           TFlt groundTruthAlpha = EI.GetDat().GetDat(groundTruthStep);
           mse += TMath::Sqr(groundTruthAlpha);
           mae += TFlt::Abs(groundTruthAlpha);
        }
     }

     int nodeSize = GroundTruth.GetNodes();
     int edgeSize = nodeSize * (nodeSize-1) / 2;
     mse /= double(edgeSize);
     mae /= double(edgeSize);
     MSE[i].AddDat(step,mse);
     MAE[i].AddDat(step,mae);
     printf("%s MSE: %f, MAE: %f\n", ModelNames[i](), mse(), mae());
  }
}

void Evaluator::PlotPRC(const TStr &OutFNm) const {

   if (PRC.Len() < 1) return;
   TVec<TFlt> steps; PRC[0].GetKeyV(steps);

   for (int i=0;i<steps.Len();i++) {
      TFlt step = steps[i];
      TStr outFNm = TStr::Fmt("%s-PRC-%f",OutFNm.CStr(),step());
      TGnuPlot plot(outFNm,"PRC Curve");
      plot.SetXLabel("Recall"); 
      plot.SetYLabel("Precision");
 
      for (int j=0;j<PRC.Len();j++) {
         TInt pointType = 2*(j+1);
         TStr style = "pt ";
         style+=pointType.GetStr();
         style+=" ps 2"; 
         plot.AddPlot(PRC[j].GetDat(step), gpwLinesPoints, ModelNames[j], style);
      }
      plot.SavePng(outFNm+".png", 1000, 800, "", "set terminal png size 1000,800 font \"LiberationMono-Regular,20\""); 
   }
}

void Evaluator::PlotMSE(const TStr &OutFNm) const {
   if (MSE.Len()<1) return;
   TFltV steps; MSE[0].GetKeyV(steps);

   for (int i=0;i<steps.Len();i++) {
      TFlt step = steps[i];
      TStr outFNm = TStr::Fmt("%s-MSE-%f",OutFNm.CStr(),step());
      TGnuPlot plot(outFNm,"MSE Bar Chart");
      plot.SetYLabel("MSE"); 
      plot.AddCmd("set boxwidth 0.5");
      plot.AddCmd("unset xtics");
      TFlt maxYValue = -1.0;
      for (int j=0;j<MSE.Len();j++) {
         TFltV xValue; xValue.Add((j+1.0)*0.5);
         TFltV yValue; yValue.Add(MSE[j].GetDat(step));
         if (yValue[0] > maxYValue) maxYValue = yValue[0];
         plot.AddPlot(xValue, yValue, gpwBoxes, ModelNames[j], "fill solid 1");
      }
      plot.SetYRange(0, maxYValue+2.0);
      plot.SetXRange(0, 0.5*( 1.0 + MSE.Len())+2.0);
      plot.SavePng(outFNm+".png", 1000, 800, "", "set terminal png size 1000,800 font \"LiberationMono-Regular,20\""); 
   }
}

TVec<TFlt> Evaluator::GetSteps(size_t i) const {
   IAssert(i < (size_t)InferredNetworks.Len());
   TVec<TFlt> steps;
   InferredNetworks[i].BegEI().GetDat().GetKeyV(steps);
   return steps;
}

TFlt Evaluator::GetGroundTruthTimeStep(TFlt step) const {
   if (GroundTruth.GetEdges()==0) {
      printf("Ground truth network has no edge!!\n");
      exit(0);
   }
   TVec<TFlt> steps; GroundTruth.BegEI().GetDat().GetKeyV(steps);
   TFlt groundTruthStep;

   for (int t=1;t<steps.Len();t++) {
      if (step < steps[t]) {
         groundTruthStep = steps[t-1];
         break;
      }
   }
   return groundTruthStep;
}

TFlt Evaluator::GetInferredTimeStep(TFlt step, size_t i) const {
   TVec<TFlt> steps = GetSteps(i);
   TFlt InferredTimeStep = steps[0];

   for (int t=1;t<steps.Len();t++) {
      if (step < steps[t]) {
         InferredTimeStep = steps[t-1];
         break;
      }
   }
   return InferredTimeStep;
}
