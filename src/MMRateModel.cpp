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

void MMRateModel::Init() {
   for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
      InferredNetwork.AddNode(NI.GetKey(), NI.GetDat().Name);
   }
}

void MMRateModel::Infer(const TFltV& Steps) {
   
   mMRateFunctionConfigure.configure.shapingFunction = new EXPShapingFunction();
   lossFunction.set(mMRateFunctionConfigure);
   em.set(eMConfigure);
   TIntFltH CascadesIdx;
   Data data = {nodeInfo.NodeNmH, CascH, CascadesIdx, 0.0};
   lossFunction.InitLatentVariable(data, eMConfigure);
   
   TSampling Sampling = eMConfigure.pGDConfigure.sampling;
   TStrV ParamSamplingV; eMConfigure.pGDConfigure.ParamSampling.SplitOnAllCh(';', ParamSamplingV);

   for (int t=1; t<Steps.Len(); t++) {
      TIntFltH CascadesIdx;
      for (int i=0; i<CascH.Len(); i++) {
         if (CascH[i].LenBeforeT(Steps[t]) > 1 &&
            ( (Sampling!=WIN_SAMPLING && Sampling!=WIN_EXP_SAMPLING) ||
              (Sampling==WIN_SAMPLING && (Steps[t]-CascH[i].GetMinTm()) <= ParamSamplingV[0].GetFlt()) ||
              (Sampling==WIN_EXP_SAMPLING && (Steps[t]-CascH[i].GetMinTm()) <= ParamSamplingV[0].GetFlt()) )) {
            CascadesIdx.AddDat(i) = CascH[i].GetMinTm();
         }
      }
      Data data = {nodeInfo.NodeNmH, CascH, CascadesIdx, Steps[t]};
      em.Optimize(lossFunction, data);

      const THash<TInt,AdditiveRiskFunction>& kAlphas = lossFunction.getKAlphas();
      const THash<TInt,TFlt>& kPi = lossFunction.getKPi();

      printf("MMRate prior probability\n");
      for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) printf("topic %d probability: %f, ", piI.GetKey()(), piI.GetDat()());
      printf("\n");

      for (THash<TInt,AdditiveRiskFunction>::TIter NI = kAlphas.BegI(); !NI.IsEnd(); NI++) {
         TInt key = NI.GetKey();
         const TFlt multiplier = NI.GetDat().getParameter().getMultiplier();
         const THash<TIntPr, TFlt>& alphas = NI.GetDat().getParameter().getAlphas();
         TStrFltFltHNEDNet& inferredNetwork = InferredNetwork;

         int i=0;
         for (THash<TIntPr, TFlt>::TIter AI = alphas.BegI(); !AI.IsEnd(); AI++,i++) {
            if (i%100000==0) printf("add kAlphas: %d, alphas length: %d, alpha index: %d\n", NI.GetKey()(),alphas.Len(),i);
            TInt srcNId = AI.GetKey().Val1, dstNId = AI.GetKey().Val2;
 
            TFlt alpha = AI.GetDat() * multiplier;
            if (inferredNetwork.IsEdge(srcNId, dstNId) && inferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t-1]) && 
                alpha == inferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t-1]))
               alpha = alpha * Aging;
            
            if (alpha <= mMRateFunctionConfigure.configure.MinAlpha) continue;
            if (!inferredNetwork.IsEdge(srcNId, dstNId)) inferredNetwork.AddEdge(srcNId, dstNId, TFltFltH());
 
            if (!inferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t])) inferredNetwork.GetEDat(srcNId,dstNId).AddDat(Steps[t]) = alpha;
            else if (InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) < alpha) InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) = alpha;
         }
      }   
   }
   delete mMRateFunctionConfigure.configure.shapingFunction;
}
