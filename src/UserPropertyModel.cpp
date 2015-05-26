#include <UserPropertyModel.h>
#include <kronecker.h>
#include <InfoPathFileIO.h>

void UserPropertyModel::LoadCascadesTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadCascadesTxt(FIn, CascH, nodeInfo);
}

void UserPropertyModel::LoadGroundTruthTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadNetworkTxt(FIn, Network, nodeInfo);
}

void UserPropertyModel::SaveInferred(const TStr& OutFNm) {
   InfoPathFileIO::SaveNetwork(OutFNm, InferredNetwork, nodeInfo, edgeInfo);
}

void UserPropertyModel::GenCascade(TCascade& C) {
	bool verbose = false;
	TIntFltH InfectedNIdH; TIntH InfectedBy;
	double GlobalTime, InitTime;
	double alpha;
	int StartNId;

	if (Network.GetNodes() == 0)
		return;

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

                TInt topic;
                TFlt p = TFlt::Rnd.GetUniDev(), accP = 0.0;
                for (TInt t=0; t<userPropertyFunctionConfigure.topicSize;t++) {
                   accP += lossFunction.getParameter().kPi.GetDat(t);
                   if (p <= accP) {
                      topic = t;
                      break;
                   } 
                }

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
				else alpha = (double)lossFunction.GetAlpha(NId,DstNId,topic);
				if (verbose) { printf("GlobalTime:%f, nodes:%d->%d, alpha:%f\n", GlobalTime, NId, DstNId, alpha); }

				if (alpha<1e-9) { continue; }

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

void UserPropertyModel::GenerateGroundTruth(const int& TNetwork, const int& NNodes, const int& NEdges, const TStr& NetworkParams) {
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
	  for (TNGraph::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) { 
             Network.AddNode(NI.GetId()); 
             nodeInfo.NodeNmH.AddDat(NI.GetId(), TNodeInfo(TStr::Fmt("%d", NI.GetId()), 0)); 
          }
	  for (TNGraph::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) { 
             if (EI.GetSrcNId()==EI.GetDstNId()) { continue; } 
             Network.AddEdge(EI.GetSrcNId(),EI.GetDstNId(),TFltFltH()); 
          }

	  if (verbose) { printf("Network structure has been generated succesfully!\n"); }
}

void UserPropertyModel::SaveGroundTruth() {
   UserPropertyParameter& parameter = lossFunction.getParameter();

   printf("ground truth\n");
   printf("prior probability:");
   THash<TInt,TFlt>& kPi = lossFunction.getParameter().kPi;
   for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) printf("topic %d: %f, ", piI.GetKey()(), piI.GetDat()());
   printf("\n");

   for (TStrFltFltHNEDNet::TEdgeI EI = Network.BegEI(); EI < Network.EndEI(); EI++) {
      TIntPr receiverIndex, spreaderIndex;
      receiverIndex.Val1 = EI.GetDstNId(); spreaderIndex.Val1 = EI.GetSrcNId();

      printf("%d,%d , property value:%f, acquaitance:%f, \n", \
             EI.GetSrcNId(),EI.GetDstNId(),lossFunction.GetPropertyValue(EI.GetSrcNId(),EI.GetDstNId())(),lossFunction.GetAcquaitance(EI.GetSrcNId(),EI.GetDstNId())());
      TFlt maxTopicValue = -DBL_MAX;
      TInt topic = -1;         
      for (TInt latentVariable=0; latentVariable<userPropertyFunctionConfigure.topicSize; latentVariable++) {
         receiverIndex.Val2 = spreaderIndex.Val2 = latentVariable; 
         TFlt spreaderValue = userPropertyFunctionConfigure.topicInitValue, receiverValue = userPropertyFunctionConfigure.topicInitValue;
         if (parameter.topicSpread.IsKey(spreaderIndex)) spreaderValue = parameter.topicSpread.GetDat(spreaderIndex);
         if (parameter.topicReceive.IsKey(receiverIndex)) receiverValue = parameter.topicReceive.GetDat(receiverIndex);
         TFlt topicValue = spreaderValue * receiverValue;
         if (topicValue > maxTopicValue) {
            maxTopicValue = topicValue;
            topic = latentVariable;
         }
         printf("\t\ttopic %d alpha:%f, topic spread value:%f, topic receiver value:%f, \n",\
                latentVariable(),lossFunction.GetAlpha(EI.GetSrcNId(),EI.GetDstNId(),latentVariable)(), spreaderValue(), receiverValue());
      }
      //printf("\n");
      EI.GetDat().AddDat(0.0,lossFunction.GetAlpha(EI.GetSrcNId(),EI.GetDstNId(),topic));
   }
}

void UserPropertyModel::Init() {
   for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
      InferredNetwork.AddNode(NI.GetKey(), NI.GetDat().Name);
   }
}

extern TFlt sigmoid(TFlt t);

void UserPropertyModel::Infer(const TFltV& Steps) {
   
   userPropertyFunctionConfigure.shapingFunction = new EXPShapingFunction();
   lossFunction.set(userPropertyFunctionConfigure);
   em.set(eMConfigure);
   TIntFltH CascadesIdx;
   Data data = {nodeInfo.NodeNmH, CascH, CascadesIdx, 0.0};
   lossFunction.InitLatentVariable(data, eMConfigure);
   lossFunction.initParameter(data, userPropertyFunctionConfigure);
   
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

      const THash<TIntPr,TFlt>& acquaintance     = lossFunction.getParameter().acquaintance;
      const THash<TIntPr,TFlt>& receiverProperty = lossFunction.getParameter().receiverProperty;
      const THash<TIntPr,TFlt>& spreaderProperty = lossFunction.getParameter().spreaderProperty;
      const THash<TIntPr,TFlt>& topicReceive = lossFunction.getParameter().topicReceive;
      const THash<TIntPr,TFlt>& topicSpread = lossFunction.getParameter().topicSpread;
      const TFlt multiplier = lossFunction.getParameter().multiplier;
   
      printf("prior probability:");
      const THash<TInt,TFlt>& kPi = lossFunction.getParameter().kPi;
      for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) printf("topic %d: %f, ", piI.GetKey()(), piI.GetDat()());
      printf("\n");

      int i=0;
      int nodeSize = nodeInfo.NodeNmH.Len();
      for (THash<TInt, TNodeInfo>::TIter SI = nodeInfo.NodeNmH.BegI(); !SI.IsEnd(); SI++) {
         for (THash<TInt, TNodeInfo>::TIter DI = nodeInfo.NodeNmH.BegI(); !DI.IsEnd(); DI++,i++) {
            if (SI.GetKey()==DI.GetKey()) continue;

            TIntPr srcIndex, dstIndex, acquaintanceIndex;
            srcIndex.Val1 = SI.GetKey(); dstIndex.Val1 = DI.GetKey();
            acquaintanceIndex.Val1 = SI.GetKey(); acquaintanceIndex.Val2 = DI.GetKey();
            TFlt acquaintanceValue = userPropertyFunctionConfigure.acquaintanceInitValue;
            if (acquaintance.IsKey(acquaintanceIndex)) acquaintanceValue = acquaintance.GetDat(acquaintanceIndex);
            //if (acquaintanceValue <= edgeInfo.MinAlpha) continue;

            TFlt alpha = 0.0;
            for (TInt index=0; index < userPropertyFunctionConfigure.propertySize; index++) {
               srcIndex.Val2 = dstIndex.Val2 = index;
               TFlt srcValue = userPropertyFunctionConfigure.propertyInitValue, dstValue = userPropertyFunctionConfigure.propertyInitValue;
               if (spreaderProperty.IsKey(srcIndex)) srcValue = spreaderProperty.GetDat(srcIndex);
               if (receiverProperty.IsKey(dstIndex)) dstValue = receiverProperty.GetDat(dstIndex);
               alpha += srcValue * dstValue;
            }
            alpha /= (TFlt) userPropertyFunctionConfigure.propertySize;

            TInt srcNId = SI.GetKey(), dstNId = DI.GetKey();
            printf("%d,%d: property value:%f, acquaintance value:%f, \n", srcNId(), dstNId(), alpha(), acquaintanceValue());
            TFlt topicValue = -DBL_MAX; TInt topic = -1;
            for (TInt latentVariable = 0; latentVariable < userPropertyFunctionConfigure.topicSize; latentVariable++) {
               srcIndex.Val2 = dstIndex.Val2 = latentVariable;
               TFlt srcValue = userPropertyFunctionConfigure.topicInitValue, dstValue = userPropertyFunctionConfigure.topicInitValue;
               if (topicSpread.IsKey(srcIndex))  srcValue = topicSpread.GetDat(srcIndex);
               if (topicReceive.IsKey(dstIndex)) dstValue = topicReceive.GetDat(dstIndex);
               if (srcValue * dstValue > topicValue && kPi.GetDat(latentVariable) > 0.0001) { 
                  topicValue = srcValue * dstValue;
                  topic = latentVariable;
               }
               printf("\t\ttopic %d alpha:%f, topic spreader value:%f, topic receiver value:%f\n", \
                      latentVariable(), acquaintanceValue * edgeInfo.MaxAlpha * sigmoid(alpha + srcValue * dstValue),srcValue(),dstValue());
            }
            //printf("\n");

            //TInt srcNId = SI.GetKey(), dstNId = DI.GetKey();
            //if (i%100000==0) printf("add edge: %d,%d , edge size: %d, edge index: %d\n", srcNId(), dstNId(), nodeSize*nodeSize,i);
            TFlt propertyValue = alpha;
            alpha += topicValue;
            alpha *= multiplier;
            alpha = acquaintanceValue * edgeInfo.MaxAlpha * sigmoid(alpha);
            //printf("%d,%d: alpha:%f, acquaintance value:%f, multiplier:%f, property value:%f, topic:%d, topic value:%f\n",\
                   srcNId(), dstNId(), alpha(), acquaintanceValue(), multiplier(), propertyValue(), topic(), topicValue());

            if (InferredNetwork.IsEdge(srcNId, dstNId) && InferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t-1]) && 
                alpha == InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t-1]))
               alpha = alpha * Aging;
            
            if (alpha <= edgeInfo.MinAlpha) continue;
            if (alpha > edgeInfo.MaxAlpha) alpha = edgeInfo.MaxAlpha;
            if (!InferredNetwork.IsEdge(srcNId, dstNId)) InferredNetwork.AddEdge(srcNId, dstNId, TFltFltH());
 
            if (!InferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t])) InferredNetwork.GetEDat(srcNId,dstNId).AddDat(Steps[t]) = alpha;
         }
      }   
   }
   delete userPropertyFunctionConfigure.shapingFunction;
}
