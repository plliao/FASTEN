#include <SoftMixCascadesModel.h>
#include <kronecker.h>
#include <InfoPathFileIO.h>
#include <cmath>

void SoftMixCascadesModel::LoadCascadesTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadCascadesTxt(FIn, CascH, nodeInfo);
}

void SoftMixCascadesModel::LoadGroundTruthTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadNetworkTxt(FIn, Network, nodeInfo);
}

void SoftMixCascadesModel::SaveInferred(const TStr& OutFNm) {
   InfoPathFileIO::SaveNetwork(OutFNm, InferredNetwork, nodeInfo, edgeInfo);
}

void SoftMixCascadesModel::SaveWeights(const TStr& OutFNm) {
   TFOut FOut(OutFNm);
   THash<TInt, THash<TInt,TFlt> >& cascadesWeights = lossFunction.parameter.cascadesWeights;
   for (THash<TInt, THash<TInt,TFlt> >::TIter WI = cascadesWeights.BegI(); !WI.IsEnd(); WI++) {
      THash<TInt,TFlt>& weight = WI.GetDat();
      FOut.PutStr(TStr::Fmt("%d;", WI.GetKey()));

      TInt size = weight.Len();
      for (THash<TInt,TFlt>::TIter VI = weight.BegI(); !VI.IsEnd(); VI++) {
         FOut.PutStr(TStr::Fmt("%f",VI.GetDat()));
         if (VI.GetKey()!=size-1) FOut.PutStr(",");
      }
      FOut.PutStr("\n");
   }
}

void SoftMixCascadesModel::GenCascade(TCascade& C) {
	bool verbose = false;
	TIntFltH InfectedNIdH; TIntH InfectedBy;
	double GlobalTime, InitTime;
	double alpha;
	int StartNId;

	if (Network.GetNodes() == 0)
		return;

        lossFunction.parameter.GenCascadeWeight(C);

        // set random seed
        //TInt::Rnd.Randomize();

	while (C.Len() < 2) {
		C.Clr();
		InfectedNIdH.Clr();
		InfectedBy.Clr();

		InitTime = TFlt::Rnd.GetUniDev() * TotalTime; // random starting point <TotalTime
		GlobalTime = InitTime;

		StartNId = Network.GetRndNId();
		InfectedNIdH.AddDat(StartNId) = GlobalTime;

		while (true) {
			// sort by time & get the oldest node that did not run infection
			InfectedNIdH.SortByDat(true);
			const int& NId = InfectedNIdH.BegI().GetKey();
			GlobalTime = InfectedNIdH.BegI().GetDat();

			// all the nodes has run infection
			if ( GlobalTime >= TFlt::GetMn(TotalTime, InitTime+Window) )
				break;

			// add current oldest node to the network and set its time
			C.Add(NId, GlobalTime);

			if (verbose) { printf("GlobalTime:%f, infected node:%d\n", GlobalTime, NId); }

			// run infection from the current oldest node
			TStrFltFltHNEDNet::TNodeI NI = Network.GetNI(NId);
			for (int e = 0; e < NI.GetOutDeg(); e++) {
				const int DstNId = NI.GetOutNId(e);

				// choose the current tx rate (we assume the most recent tx rate)
				if (Network.IsEdge(NId,DstNId) && Network.GetEDat(NId, DstNId).Len() > 0) {
				   TFltFltH& Alphas = Network.GetEDat(NId, DstNId);
				   for (int j=0; j<Alphas.Len() && Alphas.GetKey(j)<GlobalTime; j++) { alpha = Alphas[j]; }
                                }
				else alpha = (double)lossFunction.GetAlpha(NId, DstNId, C.CId);
				if (verbose) { printf("GlobalTime:%f, nodes:%d->%d, alpha:%f\n", GlobalTime, NId, DstNId, alpha); }

				if (alpha <= edgeInfo.MinAlpha) { continue; }

				// not infecting the parent
				if (InfectedBy.IsKey(NId) && InfectedBy.GetDat(NId).Val == DstNId)
					continue;

				double sigmaT;
				switch (nodeInfo.Model) {
				case EXP:
					// exponential with alpha parameter
					sigmaT = TInt::Rnd.GetExpDev(alpha);
					break;
				case POW:
					// power-law with alpha parameter
					sigmaT = TInt::Rnd.GetPowerDev(1+alpha);
					while (sigmaT < Delta) { sigmaT = Delta*TInt::Rnd.GetPowerDev(1+alpha); }
					break;
				case RAY:
					// rayleigh with alpha parameter
					sigmaT = TInt::Rnd.GetRayleigh(1/sqrt(alpha));
					break;
				default:
					sigmaT = 1;
					break;
				}

				IAssert(sigmaT >= 0);

				double t1 = TFlt::GetMn(GlobalTime + sigmaT, TFlt::GetMn(InitTime+Window, TotalTime));

				if (InfectedNIdH.IsKey(DstNId)) {
					double t2 = InfectedNIdH.GetDat(DstNId);
					if ( t2 > t1 && t2 < TFlt::GetMn(InitTime+Window, TotalTime)) {
						InfectedNIdH.GetDat(DstNId) = t1;
						InfectedBy.GetDat(DstNId) = NId;
					}
				} else {
					InfectedNIdH.AddDat(DstNId) = t1;
					InfectedBy.AddDat(DstNId) = NId;
				}
			}

			// we cannot delete key (otherwise, we cannot sort), so we assign a big time (InitTime + window cut-off)
			InfectedNIdH.GetDat(NId) = TFlt::GetMn(InitTime+Window, TotalTime);
		}
    }

	C.Sort();

}

void SoftMixCascadesModel::GenerateGroundTruth(const int& TNetwork, const int& NNodes, const int& NEdges, const TStr& NetworkParams) {
   TIntFltH positionHash;
   Data data = {nodeInfo.NodeNmH, CascH, positionHash, 0};
   lossFunction.set(softMixCascadesFunctionConfigure);
   lossFunction.init(data);

   for (TInt i=0; i < eMConfigure.latentVariableSize; i++) {
	  bool verbose = true;
	  PNGraph Graph;
	  TKronMtx SeedMtx;
	  TStr MtxNm;

	  // set random seed
	  //TInt::Rnd.Randomize();

	  switch (TNetwork) {
	  // 2-dimension kronecker network
	  case 0:
		  printf("Kronecker graph for Ground Truth\n");
		  SeedMtx = TKronMtx::GetMtx(NetworkParams.CStr()); // 0.5,0.5,0.5,0.5

		  printf("\n*** Seed matrix:\n");
		  SeedMtx.Dump();

		  Graph = TKronMtx::GenFastKronecker(SeedMtx, (int)TMath::Log2(NNodes), NEdges, true, 0);

		  break;

	  // forest fire network
	  case 1:
		  printf("Forest Fire graph for Ground Truth\n");
		  TStrV NetworkParamsV; NetworkParams.SplitOnAllCh(';', NetworkParamsV);

		  TFfGGen FF(true, // BurnExpFireP
					 NetworkParamsV[0].GetInt(), // StartNNodes (1)
					 NetworkParamsV[1].GetFlt(), // ForwBurnProb (0.2)
					 NetworkParamsV[2].GetFlt(), // BackBurnProb (0.17)
					 NetworkParamsV[3].GetInt(), // DecayProb (1)
					 NetworkParamsV[4].GetInt(), // Take2AmbasPrb (0)
					 NetworkParamsV[5].GetInt()); // OrphanPrb (0)

		  FF.GenGraph(NNodes, false);
		  Graph = FF.GetGraph();

		  break;
	  }

	  // fill network structure with graph
	  if (i==0) {
	     for (TNGraph::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) { 
                Network.AddNode(NI.GetId()); 
                nodeInfo.NodeNmH.AddDat(NI.GetId(), TNodeInfo(TStr::Fmt("%d", NI.GetId()), 0)); 
             }
          }
	  for (TNGraph::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) { 
             if (EI.GetSrcNId()==EI.GetDstNId()) { continue; } 
             if (!Network.IsEdge(EI.GetSrcNId(),EI.GetDstNId()))
                Network.AddEdge(EI.GetSrcNId(),EI.GetDstNId(),TFltFltH()); 
             TIntPr index(EI.GetSrcNId(),EI.GetDstNId());
             lossFunction.parameter.kAlphas.GetDat(i).AddDat(index, 0.0);
          }

	  if (verbose) { printf("Network structure has been generated succesfully!\n"); }
   }
   lossFunction.GenParameter();
}

void SoftMixCascadesModel::SaveGroundTruth(TStr fileNm) {
   printf("ground truth\n");

   for (TInt latentVariable=0; latentVariable < softMixCascadesFunctionConfigure.latentVariableSize; latentVariable++) {
      TFOut FOut(fileNm + TStr::Fmt("-%d-network.txt", latentVariable+1));
      for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
         FOut.PutStr(TStr::Fmt("%d,%s\n", NI.GetKey().Val, NI.GetDat().Name.CStr()));
      }
      FOut.PutStr("\n");
   }

   for (TStrFltFltHNEDNet::TEdgeI EI = Network.BegEI(); EI < Network.EndEI(); EI++) {
      TInt srcNId = EI.GetSrcNId(), dstNId = EI.GetDstNId();
      TIntPr index(srcNId, dstNId);

      TFlt maxValue = -DBL_MAX;
      printf("%d,%d , \n", srcNId(), dstNId());
      for (TInt latentVariable=0; latentVariable < softMixCascadesFunctionConfigure.latentVariableSize; latentVariable++) {
         THash<TIntPr, TFlt>& alphas = lossFunction.parameter.kAlphas.GetDat(latentVariable); 

         TFlt alpha = 0.0;
         if (alphas.IsKey(index)) alpha = alphas.GetDat(index);
         if (alpha > maxValue) maxValue = alpha;

         printf("\t\ttopic %d alpha:%f \n", latentVariable(), alpha());
         if (alpha > edgeInfo.MinAlpha ) {
            TFOut FOut(fileNm + TStr::Fmt("-%d-network.txt", latentVariable+1), true);
            FOut.PutStr(TStr::Fmt("%d,%d,%f,%f\n", srcNId, dstNId, 0.0, alpha));  
         }
      }
      printf("\n");
      EI.GetDat().AddDat(0.0, maxValue);
   }
}

void SoftMixCascadesModel::Init() {
   for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
      InferredNetwork.AddNode(NI.GetKey(), NI.GetDat().Name);
   }
}

void SoftMixCascadesModel::Infer(const TFltV& Steps, const TStr& OutFNm) {
  
   switch (nodeInfo.Model) {
      case POW :
         softMixCascadesFunctionConfigure.shapingFunction = new POWShapingFunction(Delta);
         break;
      case RAY :
         softMixCascadesFunctionConfigure.shapingFunction = new RAYShapingFunction();
         break;
      default :
         softMixCascadesFunctionConfigure.shapingFunction = new EXPShapingFunction(); 
   } 
   lossFunction.set(softMixCascadesFunctionConfigure);
   pgd.set(eMConfigure.pGDConfigure);
   TIntFltH CascadesPositions;
   Data data = {nodeInfo.NodeNmH, CascH, CascadesPositions, 0.0};
   lossFunction.init(data);
   lossFunction.initParameter();
  
   printf("Soft Mix Cascades initialization done\n");
   fflush(stdout);
 
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
      pgd.Optimize(lossFunction, data);

      const THash<TInt, THash<TIntPr, TFlt> >& kAlphas = lossFunction.getParameter().kAlphas;

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
            
            if (alpha <= softMixCascadesFunctionConfigure.MinAlpha) continue;
            if (!inferredNetwork.IsEdge(srcNId, dstNId)) inferredNetwork.AddEdge(srcNId, dstNId, TFltFltH());
 
            FOut.PutStr(TStr::Fmt("%d,%d,%f,%f\n", srcNId, dstNId, Steps[t], alpha));

            if (!inferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t])) inferredNetwork.GetEDat(srcNId,dstNId).AddDat(Steps[t]) = alpha;
            else if (InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) < alpha) InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t]) = alpha;
         }
      }   
   }
   delete softMixCascadesFunctionConfigure.shapingFunction;
}
