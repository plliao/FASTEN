#include <Info.h>

void Info::LoadCascadesTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadCascadesTxt(FIn, CascH, nodeInfo);
}

void Info::LoadGroundTruthTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadNetworkTxt(FIn, Network, nodeInfo);
}

void Info::SaveInferred(const TStr& OutFNm) {
   InfoPathFileIO::SaveNetwork(OutFNm, InferredNetwork, nodeInfo, edgeInfo);
}

void Info::Init() {
   for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
      InferredNetwork.AddNode(NI.GetKey(), NI.GetDat().Name);
   }
}

void Info::Infer(const TFltV& Steps) {
   
   switch (nodeInfo.Model) {
      case POW :
         additiveRiskFunctionConfigure.shapingFunction = new POWShapingFunction(Delta);
         break;
      case RAY :
         additiveRiskFunctionConfigure.shapingFunction = new RAYShapingFunction();
         break;
      default :
         additiveRiskFunctionConfigure.shapingFunction = new EXPShapingFunction(); 
   } 
   lossFunction.set(additiveRiskFunctionConfigure);
   pgd.set(pGDConfigure);
   
   TSampling Sampling = pGDConfigure.sampling;
   TStrV ParamSamplingV; pGDConfigure.ParamSampling.SplitOnAllCh(';', ParamSamplingV);

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
      pgd.Optimize(lossFunction, data);

      const THash<TIntPr, TFlt> &alphas = lossFunction.parameter.alphas;

      for (THash<TIntPr, TFlt>::TIter AI = alphas.BegI(); !AI.IsEnd(); AI++) {
         TInt srcNId = AI.GetKey().Val1, dstNId = AI.GetKey().Val2;

         TFlt alpha = AI.GetDat();
         if (alpha < edgeInfo.MinAlpha) continue;
         if (!InferredNetwork.IsEdge(srcNId, dstNId)) InferredNetwork.AddEdge(srcNId, dstNId, TFltFltH());

         if (InferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t-1]) && alpha == InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t-1]))
            alpha = alpha * Aging;
         if (alpha > edgeInfo.MaxAlpha) alpha = edgeInfo.MaxAlpha;

 
         InferredNetwork.GetEDat(srcNId,dstNId).AddDat(Steps[t]) = alpha;          
      }
      
   }
   delete additiveRiskFunctionConfigure.shapingFunction;
}
