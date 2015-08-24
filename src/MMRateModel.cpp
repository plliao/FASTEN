#include <MMRateModel.h>

void MMRateModel::LoadCascadesTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadCascadesTxt(FIn, CascH, nodeInfo);
}

void MMRateModel::LoadGroundTruthTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadNetworkTxt(FIn, Network, nodeInfo);
}

void MMRateModel::SaveInferred(const TStr& OutFNm) {
   InfoPathFileIO::SaveNetwork(OutFNm, InferredNetwork, nodeInfo, edgeInfo);
}

void MMRateModel::SaveDiffusionPatterns(const TStr& OutFNm) {
   TFOut FOut(OutFNm);
   const THash<TInt,TFlt>& diffusionPatterns = lossFunction.getParameter().diffusionPatterns;
   for (THash<TInt,TFlt>::TIter IAI = diffusionPatterns.BegI(); !IAI.IsEnd(); IAI++) {
      FOut.PutStr(TStr::Fmt("%d;%f\n",IAI.GetKey()(),IAI.GetDat()()));
   }
}

void MMRateModel::Init() {
   for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
      InferredNetwork.AddNode(NI.GetKey(), NI.GetDat().Name);
      MaxNetwork.AddNode(NI.GetKey(), NI.GetDat().Name);
   }
}

void MMRateModel::Infer(const TFltV& Steps, const TStr& OutFNm) {
  
   switch (nodeInfo.Model) {
      case POW :
         mMRateFunctionConfigure.shapingFunction = new POWShapingFunction(Delta);
         break;
      case RAY :
         mMRateFunctionConfigure.shapingFunction = new RAYShapingFunction();
         break;
      default :
         mMRateFunctionConfigure.shapingFunction = new EXPShapingFunction(); 
   } 
   lossFunction.set(mMRateFunctionConfigure);
   em.set(eMConfigure);
   TIntFltH CascadesPositions;
   Data data = {nodeInfo.NodeNmH, CascH, CascadesPositions, 0.0};
   lossFunction.InitLatentVariable(data, eMConfigure);
   
   TSampling Sampling = eMConfigure.pGDConfigure.sampling;
   TStrV ParamSamplingV; eMConfigure.pGDConfigure.ParamSampling.SplitOnAllCh(';', ParamSamplingV);

   for (int t=1; t<Steps.Len(); t++) {
      TIntFltH CascadesPositions;
      for (int i=0; i<CascH.Len(); i++) {
         if (CascH[i].LenBeforeT(Steps[t]) > 1 &&
            ( (Sampling!=WIN_SAMPLING && Sampling!=WIN_EXP_SAMPLING) ||
              (Sampling==WIN_SAMPLING && (Steps[t]-CascH[i].GetMinTm()) <= ParamSamplingV[0].GetFlt()) ||
              (Sampling==WIN_EXP_SAMPLING && (Steps[t]-CascH[i].GetMinTm()) <= ParamSamplingV[0].GetFlt()) )) {
            CascadesPositions.AddDat(i) = CascH[i].GetMinTm();
         }
      }
      Data data = {nodeInfo.NodeNmH, CascH, CascadesPositions, Steps[t]};
      lossFunction.initPotentialEdges(data);
      em.Optimize(lossFunction, data);

      const THash<TInt, THash<TIntPr, TFlt> >& kAlphas = lossFunction.getParameter().kAlphas;
      const THash<TInt,TFlt>& kPi = lossFunction.getParameter().kPi;

      printf("MMRate prior probability\n");
      for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) printf("topic %d probability: %f, ", piI.GetKey()(), piI.GetDat()());
      printf("\n");
         

      for (THash<TInt, THash<TIntPr, TFlt> >::TIter NI = kAlphas.BegI(); !NI.IsEnd(); NI++) {
         TInt key = NI.GetKey();
         const THash<TIntPr, TFlt>& alphas = NI.GetDat();
         TStrFltFltHNEDNet& inferredNetwork = InferredNetwork;

         TFOut FOut(OutFNm + TStr("_") + key.GetStr() + ".txt");

         for (THash<TInt, TNodeInfo>::TIter NodeI = nodeInfo.NodeNmH.BegI(); NodeI < nodeInfo.NodeNmH.EndI(); NodeI++) {
            FOut.PutStr(TStr::Fmt("%d,%s\n", NodeI.GetKey().Val, NodeI.GetDat().Name.CStr()));
         }
         FOut.PutStr("\n");

         int i=0;
         for (THash<TIntPr, TFlt>::TIter AI = alphas.BegI(); !AI.IsEnd(); AI++,i++) {
            if (i%100000==0) printf("add kAlphas: %d, alphas length: %d, alpha index: %d\n", NI.GetKey()(),alphas.Len(),i);
            TInt srcNId = AI.GetKey().Val1, dstNId = AI.GetKey().Val2;
 
            TFlt alpha = AI.GetDat();
            if (inferredNetwork.IsEdge(srcNId, dstNId) && inferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t-1]) && 
                alpha == inferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t-1]))
               alpha = alpha * Aging;
            
            if (alpha <= mMRateFunctionConfigure.MinAlpha) continue;
            if (!inferredNetwork.IsEdge(srcNId, dstNId)) inferredNetwork.AddEdge(srcNId, dstNId, TFltFltH());
            if (!MaxNetwork.IsEdge(srcNId, dstNId)) MaxNetwork.AddEdge(srcNId, dstNId, TFltFltH());
 
            FOut.PutStr(TStr::Fmt("%d,%d,%f,%f\n", srcNId, dstNId, Steps[t], alpha));

            if (!inferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t])) inferredNetwork.GetEDat(srcNId,dstNId).AddDat(Steps[t]) = alpha * kPi.GetDat(key);
            else InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) += alpha * kPi.GetDat(key);

            if (!MaxNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t])) MaxNetwork.GetEDat(srcNId,dstNId).AddDat(Steps[t]) = alpha;
            else if (MaxNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) < alpha) MaxNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) = alpha;
         }
      }   
   }
   InfoPathFileIO::SaveNetwork(OutFNm + "_Max.txt", MaxNetwork, nodeInfo, edgeInfo);
   delete mMRateFunctionConfigure.shapingFunction;
}
