#include <MixCascadesModel.h>

void MixCascadesModel::LoadCascadesTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadCascadesTxt(FIn, CascH, nodeInfo);
}

void MixCascadesModel::LoadGroundTruthTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadNetworkTxt(FIn, Network, nodeInfo);
}

void MixCascadesModel::SaveInferred(const TStr& OutFNm) {
   InfoPathFileIO::SaveNetwork(OutFNm, InferredNetwork, nodeInfo, edgeInfo);
}

void MixCascadesModel::Init() {
   for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
      InferredNetwork.AddNode(NI.GetKey(), NI.GetDat().Name);
   }
}

void MixCascadesModel::Infer(const TFltV& Steps, const TStr& OutFNm) {
  
   switch (nodeInfo.Model) {
      case POW :
         mixCascadesFunctionConfigure.configure.shapingFunction = new POWShapingFunction(Delta);
         break;
      case RAY :
         mixCascadesFunctionConfigure.configure.shapingFunction = new RAYShapingFunction();
         break;
      default :
         mixCascadesFunctionConfigure.configure.shapingFunction = new EXPShapingFunction(); 
   } 
   lossFunction.set(mixCascadesFunctionConfigure);
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
      em.Optimize(lossFunction, data);

      const THash<TInt,AdditiveRiskFunction>& kAlphas = lossFunction.getKAlphas();
      const THash<TInt,TFlt>& kPi = lossFunction.getKPi();

      printf("MixCascades prior probability\n");
      for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) printf("topic %d probability: %f, ", piI.GetKey()(), piI.GetDat()());
      printf("\n");
         

      for (THash<TInt,AdditiveRiskFunction>::TIter NI = kAlphas.BegI(); !NI.IsEnd(); NI++) {
         TInt key = NI.GetKey();
         const TFlt multiplier = NI.GetDat().getParameter().getMultiplier();
         const THash<TIntPr, TFlt>& alphas = NI.GetDat().getParameter().getAlphas();
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
 
            TFlt alpha = AI.GetDat() * multiplier;
            if (inferredNetwork.IsEdge(srcNId, dstNId) && inferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t-1]) && 
                alpha == inferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t-1]))
               alpha = alpha * Aging;
            
            if (alpha <= mixCascadesFunctionConfigure.configure.MinAlpha) continue;
            if (!inferredNetwork.IsEdge(srcNId, dstNId)) inferredNetwork.AddEdge(srcNId, dstNId, TFltFltH());
 
            FOut.PutStr(TStr::Fmt("%d,%d,%f,%f\n", srcNId, dstNId, Steps[t], alpha));

            if (!inferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t])) inferredNetwork.GetEDat(srcNId,dstNId).AddDat(Steps[t]) = alpha;
            else if (InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) < alpha) InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) = alpha;
         }
      }   
   }
   delete mixCascadesFunctionConfigure.configure.shapingFunction; 
}
