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
   const TFlt scale = (maxAlpha-minAlpha)/PRCPointNm;
   TFlt groundTruthTimeStep = GetGroundTruthTimeStep(step); 
   TIntV NIdV; GroundTruth.GetNIdV(NIdV);
   int nodeSize = NIdV.Len();

   for (int i=0;i<InferredNetworks.Len();i++) {
      PRC.Add(DyPRCPoints());
      DyPRCPoints &dyPRCPoints = PRC[i];
      if (dyPRCPoints.IsKey(step)) continue;
     
      dyPRCPoints.AddDat(step,PRCPoints());
      PRCPoints &prcPoints = dyPRCPoints.GetDat(step);
      prcPoints.Reserve(PRCPointNm);

      printf("Evluating PRC points, model:%s\n",ModelNames[i]());

      for (int j=0;j<PRCPointNm;j++) {

         if (j!=0 && prcPoints.Last().Val1==0.0 && prcPoints.Last().Val2==0.0) break;

         PRCPoint prcPoint;
         TFlt threshold = minAlpha + j * scale;
         int TP = 0, FP = 0, TN = 0, FN = 0;

         TStrFltFltHNEDNet &inferredNetwork = InferredNetworks[i];

         #pragma omp parallel for
         for (int src=0; src<nodeSize; src++) {
            int srcTP = 0, srcFP = 0, srcTN = 0, srcFN = 0;
            for (int dst=0; dst<nodeSize; dst++) {
               int srcNId = NIdV[src];
               int dstNId = NIdV[dst];
               if (srcNId==dstNId) continue;
               
               bool inferredEdge = false;
               bool groundTruthEdge = false;
            
               if (GroundTruth.IsEdge(srcNId,dstNId)) groundTruthEdge = true;
               if (inferredNetwork.IsEdge(srcNId,dstNId) && inferredNetwork.GetEDat(srcNId,dstNId).GetDat(step) >= threshold) inferredEdge = true;
               else if (0.0 >= threshold) inferredEdge = true;

               if (inferredEdge && groundTruthEdge) srcTP++;
               else if (inferredEdge && !groundTruthEdge) srcFP++;
               else if (!inferredEdge && groundTruthEdge) srcFN++;
               else srcTN++;
            }

            #pragma omp critical
            {
               TP += srcTP;
               FP += srcFP;
               FN += srcFN;
               TN += srcTN;
            }
         }

         if ((FN+TP)==0) prcPoint.Val1 = 0.0;
         else prcPoint.Val1 = double(TP) / double(FN + TP);
         if ((TP+FP)==0) prcPoint.Val2 = 0.0;
         else prcPoint.Val2 = double(TP) / double(TP + FP);

         printf("Threshold: %f, Precision: %f, Recall: %f, x,y: %f,%f\n", threshold(), prcPoint.Val2(), prcPoint.Val1(), prcPoint.Val1(), prcPoint.Val2());
         prcPoints.Add(prcPoint);
      }
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

     for (TStrFltFltHNEDNet::TEdgeI EI = inferredNetwork.BegEI();EI!=inferredNetwork.EndEI();EI++) {
        int srcNId = EI.GetSrcNId();
        int dstNId = EI.GetDstNId();
        if (srcNId==dstNId) continue;
        
        TFlt inferredAlpha = EI.GetDat().GetDat(step);
        
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

