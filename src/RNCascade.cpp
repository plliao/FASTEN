#include <RNCascade.h>

void RNCascade::LoadMultipleGroundTruthTxt(TSIn &SIn) {
  bool verbose = false;
  TStr Line;

  // add nodes
  while (!SIn.Eof()) {
    SIn.GetNextLn(Line);
    if (Line=="") { break; }
    TStrV NIdV; Line.SplitOnAllCh(',', NIdV);
    if (!Network.IsNode(NIdV[0].GetInt()))
       Network.AddNode(NIdV[0].GetInt(), NIdV[1]);
    if (!IsNodeNm(NIdV[0].GetInt())) {
      AddNodeNm(NIdV[0].GetInt(), TNodeInfo(NIdV[1], 0));
      DomainsIdH.AddDat(NIdV[1]) = NIdV[0].GetInt();
    }
  }

  // add edges
  while (!SIn.Eof()) {
    SIn.GetNextLn(Line);
    TStrV FieldsV; Line.SplitOnAllCh(',', FieldsV);

    TFltFltH Alphas;
    if (FieldsV.Len() == 3) { 
    Alphas.AddDat(0.0) = FieldsV[2].GetFlt(); 
    } else {
      for (int i=2; i<FieldsV.Len()-1; i+=2) {
        Alphas.AddDat(FieldsV[i].GetFlt()) = FieldsV[i+1].GetFlt();
      }
    }
    
    if (!Network.IsEdge(FieldsV[0].GetInt(), FieldsV[1].GetInt()))
       Network.AddEdge(FieldsV[0].GetInt(), FieldsV[1].GetInt(), Alphas);
    else {
       for (TFltFltH::TIter iter=Alphas.BegI();!iter.IsEnd();iter++) {
          TFlt key = iter.GetKey();
          TFltFltH &edgeDat = Network.GetEDat(FieldsV[0].GetInt(), FieldsV[1].GetInt()); 
          if (edgeDat.IsKey(key))
             edgeDat.GetDat(key) = TFlt::GetMx(edgeDat.GetDat(key),iter.GetDat());
          else 
             edgeDat.AddDat(key,iter.GetDat());
       }
    }

    if (verbose) {
      printf("Edge %d -> %d: ", FieldsV[0].GetInt(), FieldsV[1].GetInt());
      TFltFltH &AlphasE = Network.GetEDat(FieldsV[0].GetInt(), FieldsV[1].GetInt());
      for (int i=0; i<AlphasE.Len(); i+=2) { printf("(%f, %f)", AlphasE.GetKey(i).Val, AlphasE[i].Val); }
      printf("\n");
    }
  }

  printf("groundtruth nodes:%d edges:%d\n", Network.GetNodes(), Network.GetEdges());


}

void RNCascade::LoadMultipleCascadesTxt(TSIn& SIn) {
  TStr Line;
  while (!SIn.Eof()) {
    SIn.GetNextLn(Line);
    if (Line=="") { break; }
    TStrV NIdV; Line.SplitOnAllCh(',', NIdV);
    if (!NodeNmH.IsKey(NIdV[0].GetInt()))
       AddNodeNm(NIdV[0].GetInt(), TNodeInfo(NIdV[1], 0)); 
  }
  printf("All nodes read!\n");
  while (!SIn.Eof()) { 
   SIn.GetNextLn(Line); 
   int CId = CascH.Len();

   // support cascade id if any
   TStrV FieldsV; Line.SplitOnAllCh(';', FieldsV);

   // read nodes
   TStrV NIdV; FieldsV[FieldsV.Len()-1].SplitOnAllCh(',', NIdV);
   TCascade C(CId, Model);
   for (int i = 0; i < NIdV.Len(); i+=2) {
      int NId;
      double Tm; 
      NId = NIdV[i].GetInt();
      Tm = NIdV[i+1].GetFlt();
      GetNodeInfo(NId).Vol = GetNodeInfo(NId).Vol + 1;
      C.Add(NId, Tm);
    }
    C.Sort();

    AddCasc(C);

  }

  printf("All cascades read!\n");
}

void RNCascade::Reset() {
   TNIBs::Reset();
   for (TVec<TIntFltH>::TIter iter=KAveDiffAlphas.BegI();iter!=KAveDiffAlphas.EndI();iter++) {
      iter->Clr();
   }
}

void RNCascade::Init(const TFltV& Steps) {
   TNIBs::Init(Steps);
   IAssert(RNNumber>0);
   IAssert(EntropyThreshold>0.0);

   GenerateRNNIds();

   MultipleInferredNetworks.Reserve(RNNumber+1);
   KAveDiffAlphas.Reserve(RNNumber+1);

   for (int i=0;i<=RNNumber;i++) {
      KAveDiffAlphas.Add(TIntFltH());
      MultipleInferredNetworks.Add(TStrFltFltHNEDNet());
  
      for (THash<TInt, TNodeInfo>::TIter NI = NodeNmH.BegI(); NI < NodeNmH.EndI(); NI++) {
        MultipleInferredNetworks[i].AddNode(NI.GetKey(), NI.GetDat().Name);
      }
   }
}

void RNCascade::SG(const int& NId, const int& Iters, const TFltV& Steps, const TSampling& Sampling, const TStr& ParamSampling, const bool& PlotPerformance) {
  bool verbose = false;
  int currentCascade = -1;
  TIntIntH SampledCascades;
  TStrV ParamSamplingV; ParamSampling.SplitOnAllCh(';', ParamSamplingV);

  Reset();

  if (verbose) printf("Node %d\n", NId);

  // traverse through all times
  for (int t=1; t<Steps.Len(); t++) {
    // find cascades whose two first infections are earlier than Steps[t]
    TIntFltH CascadesIdx;
    int num_infections = 0;
    for (int i=0; i<CascH.Len(); i++) {
      if (CascH[i].LenBeforeT(Steps[t]) > 1 &&
        ( (Sampling!=WIN_SAMPLING && Sampling!=WIN_EXP_SAMPLING) ||
          (Sampling==WIN_SAMPLING && (Steps[t]-CascH[i].GetMinTm()) <= ParamSamplingV[0].GetFlt()) ||
          (Sampling==WIN_EXP_SAMPLING && (Steps[t]-CascH[i].GetMinTm()) <= ParamSamplingV[0].GetFlt()) )) {
        num_infections += CascH[i].LenBeforeT(Steps[t]);
        CascadesIdx.AddDat(i) = CascH[i].GetMinTm();
      }
    }

    // if there are not recorded cascades by Steps[t], continue
    if (CascadesIdx.Len()==0) {
      printf("WARNING: No cascades recorded by %f!\n", Steps[t].Val);
      if (PlotPerformance) { ComputePerformanceNId(NId, t, Steps); }
      continue;
    }

    // later cascades first
    CascadesIdx.SortByDat(false);

    if (verbose) printf("Solving step %f: %d cascades, %d infections\n", Steps[t].Val, CascadesIdx.Len(), num_infections);
    SampledCascades.Clr();

    // sampling cascades with no replacement
    for (int i=0; i < Iters; i++) {
      switch (Sampling) {
        case UNIF_SAMPLING:
          currentCascade = TInt::Rnd.GetUniDevInt(CascadesIdx.Len());
          break;

        case WIN_SAMPLING:
          currentCascade = TInt::Rnd.GetUniDevInt(CascadesIdx.Len());
          break;

        case EXP_SAMPLING:
          do {
            currentCascade = (int)TFlt::Rnd.GetExpDev(ParamSamplingV[0].GetFlt());
          } while (currentCascade > CascadesIdx.Len()-1);
          break;

        case WIN_EXP_SAMPLING:
          do {
            currentCascade = (int)TFlt::Rnd.GetExpDev(ParamSamplingV[1].GetFlt());
          } while (currentCascade > CascadesIdx.Len()-1);
          break;

        case RAY_SAMPLING:
          do {
            currentCascade = (int)TFlt::Rnd.GetRayleigh(ParamSamplingV[0].GetFlt());
          } while (currentCascade > CascadesIdx.Len()-1);
          break;
      }

      if (!SampledCascades.IsKey(currentCascade)) { SampledCascades.AddDat(currentCascade) = 0; }
      SampledCascades.GetDat(currentCascade)++;

      if (verbose) { printf("Cascade %d sampled!\n", currentCascade); }

      // sampled cascade
      TCascade &Cascade = CascH[CascadesIdx.GetKey(currentCascade)];

      // update gradient and alpha's
      THash<TInt,TIntPrV> KAlphasToUpdate;
      UpdateDiff(OSG, NId, Cascade, KAlphasToUpdate, Steps[t]);

      // update alpha's
      for (THash<TInt,TIntPrV>::TIter iter=KAlphasToUpdate.BegI(); !iter.IsEnd(); iter++) {
         const int k = iter.GetKey();
         const TIntPrV &AlphasToUpdate = iter.GetDat();
         for (int j=0; j<AlphasToUpdate.Len(); j++) {
           if (MultipleInferredNetworks[k].IsEdge(AlphasToUpdate[j].Val1, AlphasToUpdate[j].Val2) &&
               MultipleInferredNetworks[k].GetEDat(AlphasToUpdate[j].Val1, AlphasToUpdate[j].Val2).IsKey(Steps[t])
             ) {
                MultipleInferredNetworks[k].GetEDat(AlphasToUpdate[j].Val1, AlphasToUpdate[j].Val2).GetDat(Steps[t]) -=
                    (Gamma * KAveDiffAlphas[k].GetDat(AlphasToUpdate[j].Val1)
                     - (Regularizer==1? Mu*MultipleInferredNetworks[k].GetEDat(AlphasToUpdate[j].Val1, AlphasToUpdate[j].Val2).GetDat(Steps[t]) : 0.0));
   
                // project into alpha >= 0
                if (MultipleInferredNetworks[k].GetEDat(AlphasToUpdate[j].Val1, AlphasToUpdate[j].Val2).GetDat(Steps[t]) < Tol) {
                  MultipleInferredNetworks[k].GetEDat(AlphasToUpdate[j].Val1, AlphasToUpdate[j].Val2).GetDat(Steps[t]) = Tol;
                }
                //printf("update gradient: %f\n",Gamma * KAveDiffAlphas[k].GetDat(AlphasToUpdate[j].Val1));
   
                // project into alpha <= MaxAlpha
                if (MultipleInferredNetworks[k].GetEDat(AlphasToUpdate[j].Val1, AlphasToUpdate[j].Val2).GetDat(Steps[t]) > MaxAlpha) {
                  MultipleInferredNetworks[k].GetEDat(AlphasToUpdate[j].Val1, AlphasToUpdate[j].Val2).GetDat(Steps[t]) = MaxAlpha;
                }
             }
         }
         if (verbose) { printf("%d transmission rates updated!\n", AlphasToUpdate.Len()); }
      }
    }
    if (verbose) printf("%d different cascades have been sampled for step %f!\n", SampledCascades.Len(), Steps[t].Val);

    // For alphas that did not get updated, copy last alpha value * aging factor
    int unchanged = 0;
    for (int k=0;k<MultipleInferredNetworks.Len();k++) {
       for (TStrFltFltHNEDNet::TEdgeI EI = MultipleInferredNetworks[k].BegEI(); EI < MultipleInferredNetworks[k].EndEI(); EI++) {
         if (EI().IsKey(Steps[t]) || t == 0 || !EI().IsKey(Steps[t-1])) { continue; }

         EI().AddDat(Steps[t]) = Aging*EI().GetDat(Steps[t-1]);
         unchanged++;
       }
    }
    if (verbose) { printf("%d transmission rates that did not changed were 'aged' by %f!\n", unchanged, Aging.Val); }

    // compute performance on-the-fly
    GenerateInferredNetwork(NId);
    if (PlotPerformance) { ComputePerformanceNId(NId, t, Steps);}
    //if (t==Steps.Len()-1) printf("NId: %d, precision: %f, recall: %f\n",NId,PrecisionRecall[t-1].Val2(),PrecisionRecall[t-1].Val1());
   }
}

void RNCascade::BSG(const int& NId, const int& Iters, const TFltV& Steps, const int& BatchLen, const TSampling& Sampling, const TStr& ParamSampling, const bool& PlotPerformance) {
   TNIBs::BSG(NId, Iters, Steps, BatchLen, Sampling, ParamSampling, PlotPerformance);
}

void RNCascade::FG(const int& NId, const int& Iters, const TFltV& Steps) {
   TNIBs::FG(NId, Iters, Steps);
}

void RNCascade::UpdateDiff(const TOptMethod& OptMethod, const int& NId, TCascade& Cascade, THash<TInt,TIntPrV>& KAlphasToUpdate, const double& CurrentTime) {
  IAssert(InferredNetwork.IsNode(NId));
  
  for (int k=0;k<=RNNIds.Len();k++) {

     if (k==0) {
        bool useDefaultK = true;
        for (int j=0;j<RNNIds.Len();j++) {
           const TInt &RNNId = RNNIds[j];
           if (isRNCategory(RNNId,Cascade)) {
              useDefaultK = false;
              break;
           }
        }
        if (!useDefaultK) continue;
     }
     else {
        const TInt &RNNId = RNNIds[k-1];
        if (!isRNCategory(RNNId,Cascade)) continue;
     }
  
     TIntPrV AlphasToUpdate;
     double sum = 0.0;

     // we assume cascade is sorted & iterator returns in sorted order
     if (Cascade.IsNode(NId) && Cascade.GetTm(NId) <= CurrentTime) {
       for (THash<TInt, THitInfo>::TIter NI = Cascade.BegI(); NI < Cascade.EndI(); NI++) {
         // consider only nodes that are earlier in time
         if ( (Cascade.GetTm(NId)<=NI.GetDat().Tm) ||
              (Cascade.GetTm(NId)-Delta<=NI.GetDat().Tm && Model==POW)
            ) { break; }

         TIntPr Pair(NI.GetKey(), NId);

         // if edge/alpha doesn't exist, create
         if (!MultipleInferredNetworks[k].IsEdge(Pair.Val1, Pair.Val2)) { MultipleInferredNetworks[k].AddEdge(Pair.Val1, Pair.Val2, TFltFltH()); }
         if (!MultipleInferredNetworks[k].GetEDat(Pair.Val1, Pair.Val2).IsKey(CurrentTime)) {
           MultipleInferredNetworks[k].GetEDat(Pair.Val1, Pair.Val2).AddDat(CurrentTime) = InitAlpha;
         }

         switch(Model) {
           case EXP: // exponential
             sum += MultipleInferredNetworks[k].GetEDat(Pair.Val1, Pair.Val2).GetDat(CurrentTime).Val;
             break;
           case POW: // powerlaw
             sum += MultipleInferredNetworks[k].GetEDat(Pair.Val1, Pair.Val2).GetDat(CurrentTime).Val/(Cascade.GetTm(NId)-NI.GetDat().Tm);
             break;
           case RAY: // rayleigh
             sum += MultipleInferredNetworks[k].GetEDat(Pair.Val1, Pair.Val2).GetDat(CurrentTime).Val*(Cascade.GetTm(NId)-NI.GetDat().Tm);
             break;
           default:
             sum = 0.0;
         }
       }
     }

     // we assume cascade is sorted & iterator returns in sorted order
     for (THash<TInt, THitInfo>::TIter NI = Cascade.BegI(); NI < Cascade.EndI(); NI++) {
       // only consider nodes that are earlier in time if node belongs to the cascade
       if ( Cascade.IsNode(NId) && (Cascade.GetTm(NId)<=NI.GetDat().Tm ||
           (Cascade.GetTm(NId)-Delta<=NI.GetDat().Tm && Model==POW))
          ) { break; }

       // consider infected nodes up to CurrentTime
       if (NI.GetDat().Tm > CurrentTime) { break; }

       TIntPr Pair(NI.GetKey(), NId); // potential edge

       double val = 0.0;

       if (Cascade.IsNode(NId) && Cascade.GetTm(NId) <= CurrentTime) {
         IAssert((Cascade.GetTm(NId) - NI.GetDat().Tm) > 0.0);

         switch(Model) { // compute gradients for infected
           case EXP: // exponential
             val = (Cascade.GetTm(NId) - NI.GetDat().Tm) - 1.0/sum;
             break;
           case POW: // powerlaw
             val = log((Cascade.GetTm(NId) - NI.GetDat().Tm)/Delta) - 1.0/((Cascade.GetTm(NId)-NI.GetDat().Tm)*sum);
             break;
           case RAY: // rayleigh
             val = TMath::Power(Cascade.GetTm(NId)-NI.GetDat().Tm, 2.0)/2.0 - (Cascade.GetTm(NId)-NI.GetDat().Tm)/sum;
             break;
           default:
             val = 0.0;
         }
       } else { // compute gradients for non infected
         IAssert((CurrentTime - NI.GetDat().Tm) >= 0.0);

         switch(Model) {
           case EXP: // exponential
             val = (CurrentTime-NI.GetDat().Tm);
             // if every cascade was recorded up to a maximum Window cut-off
             if ( (Window > -1) && (CurrentTime-Cascade.GetMinTm() > Window) ) { val = (Cascade.GetMinTm()+Window-NI.GetDat().Tm); }
          break;
           case POW: // power-law
             val = TMath::Mx(log((CurrentTime-NI.GetDat().Tm)/Delta), 0.0);
             // if every cascade was recorded up to a maximum Window cut-off
             if ( (Window > -1) && (CurrentTime-Cascade.GetMinTm() > Window) ) { val = TMath::Mx(log((Cascade.GetMinTm()+Window-NI.GetDat().Tm)/Delta), 0.0); }
             break;
           case RAY: // rayleigh
             val = TMath::Power(CurrentTime-NI.GetDat().Tm,2.0)/2.0;
             // if every cascade was recorded up to a maximum Window cut-off
             if ( (Window > -1) && (CurrentTime-Cascade.GetMinTm() > Window) ) { val = TMath::Power(Cascade.GetMinTm()+Window-NI.GetDat().Tm,2.0)/2.0; }
             break;
           default:
             val = 0.0;
         }
       }

       if (!KAveDiffAlphas[k].IsKey(Pair.Val1)) { KAveDiffAlphas[k].AddDat(Pair.Val1) = 0.0; }

       switch (OptMethod) {
         case OBSG:
         case OEBSG:
         case OFG:
           // update stochastic average gradient (based on batch for OBSG and OEBSG and based on all cascades for FG)
           KAveDiffAlphas[k].GetDat(Pair.Val1) += val;
           break;
         case OSG:
         case OESG:
           // update stochastic gradient (we use a single gradient due to current cascade)
           KAveDiffAlphas[k].GetDat(Pair.Val1) = val;
         default:
           break;
       }

       AlphasToUpdate.Add(Pair);
     }
     KAlphasToUpdate.AddDat(k,AlphasToUpdate); 
  }
  return;

}

void RNCascade::SaveMultipleInferred(const TStr& OutFNm, const TIntV& NIdV) {
  TFOut FOut(OutFNm);

  // write nodes to file
  for (THash<TInt, TNodeInfo>::TIter NI = NodeNmH.BegI(); NI < NodeNmH.EndI(); NI++) {
    if (NIdV.Len() > 0 && !NIdV.IsIn(NI.GetKey())) { continue; }

    FOut.PutStr(TStr::Fmt("%d,%s\r\n", NI.GetKey().Val, NI.GetDat().Name.CStr()));
  }

  FOut.PutStr("\r\n");


  for (int k=0;k<MultipleInferredNetworks.Len();k++) {
     if (k==0) FOut.PutStr("k: 0, node: default\r\n");
     else FOut.PutStr(TStr::Fmt("k: %d, node: %d\r\n",k,RNNIds[k-1]));
     // write edges to file (not allowing self loops in the network)
     for (TStrFltFltHNEDNet::TEdgeI EI = MultipleInferredNetworks[k].BegEI(); EI < MultipleInferredNetworks[k].EndEI(); EI++) {
       if (NIdV.Len() > 0 && (!NIdV.IsIn(EI.GetSrcNId()) || !NIdV.IsIn(EI.GetDstNId()))) { continue; }
       if (!NodeNmH.IsKey(EI.GetSrcNId()) || !NodeNmH.IsKey(EI.GetDstNId())) { continue; }
   
       // not allowing self loops in the Kronecker network
       if (EI.GetSrcNId() != EI.GetDstNId()) {
         if (EI().Len() > 0) {
           TStr Line; bool IsEdge = false;
           for (int i=0; i<EI().Len(); i++) {
             if (EI()[i]>MinAlpha) {
               Line += TStr::Fmt(",%f,%f", EI().GetKey(i).Val, (EI()[i] > MaxAlpha? MaxAlpha.Val : EI()[i].Val) );
               IsEdge = true;
             } else { // we write 0 explicitly
               Line += TStr::Fmt(",%f,0.0", EI().GetKey(i).Val);
             }
           }
           // if none of the alphas is bigger than 0, no edge is written
           if (IsEdge) {
             FOut.PutStr(TStr::Fmt("%d,%d", EI.GetSrcNId(), EI.GetDstNId()));
             FOut.PutStr(Line);
             FOut.PutStr("\r\n");
           }
         }
         else
           FOut.PutStr(TStr::Fmt("%d,%d,1\r\n", EI.GetSrcNId(), EI.GetDstNId()));
       }
     }
     FOut.PutStr("\r\n");
  }
}

void RNCascade::GenerateInferredNetwork(const int& NId) {
   for (int k=0;k<MultipleInferredNetworks.Len();k++) {
      const TStrFltFltHNEDNet &inferredNetwork = MultipleInferredNetworks[k];
      const TStrFltFltHNEDNet::TNode &node = inferredNetwork.GetNode(NId);
      int inEdgeNumber = node.GetInDeg();
      for (int j=0;j<inEdgeNumber;j++) {
         int srcNId = node.GetInNId(j);
         if (!InferredNetwork.IsEdge(srcNId,NId)) {
            InferredNetwork.AddEdge(srcNId,NId,inferredNetwork.GetEDat(srcNId,NId));
         }
         else {
            TFltFltH &eDat = InferredNetwork.GetEDat(srcNId,NId);
            const TFltFltH &eDatAdded = inferredNetwork.GetEDat(srcNId,NId);
            for (TFltFltH::TIter iter = eDatAdded.BegI();iter!=eDatAdded.EndI();iter++) {
               const TFlt &key = iter.GetKey();
               const TFlt &Alpha = iter.GetDat();

               if (eDat.IsKey(key)) {
                  //if (Alpha<=Tol) continue;
                  eDat.GetDat(key) = TFlt::GetMx(eDat.GetDat(key),Alpha);
               }
               else {
                  eDat.AddDat(key, Alpha); 
               }
            }
         }
      }
   }
}

void RNCascade::GenerateRNNIds() {
   THash<TInt,TFlt> occurrenceTimes, uniqueness;
   for (THash<TInt, TCascade>::TIter iter=CascH.BegI();!iter.IsEnd();iter++) {
      TCascade &cascade = iter.GetDat();
      int cascadeLen = cascade.Len();
      for  (THash<TInt, THitInfo>::TIter nodeI=cascade.NIdHitH.BegI();!nodeI.IsEnd();nodeI++) {
         TInt NId = nodeI.GetKey();

         if (!occurrenceTimes.IsKey(NId)) occurrenceTimes.AddDat(NId,0.0);
         if (!uniqueness.IsKey(NId)) uniqueness.AddDat(NId,1.0);
        
         if (isRNCategory(NId,cascade)) { 
            occurrenceTimes.GetDat(NId)++;
            uniqueness.GetDat(NId) += (cascadeLen - 1.0);
         }
      }
   } 

   THash<TInt,TFlt> RNScores;
   for (THash<TInt,TFlt>::TIter iter=occurrenceTimes.BegI();!iter.IsEnd();iter++) {
      const TInt NId = iter.GetKey();
      TFlt score;

      if (uniqueness.GetDat(NId)>1.0) score = occurrenceTimes.GetDat(NId) * 1.0/TMath::Log(uniqueness.GetDat(NId));
      else score = occurrenceTimes.GetDat(NId);

      if (!RNScores.IsKey(NId)) RNScores.AddDat(NId,score);
   }

   RNScores.SortByDat(false);
   for (THash<TInt,TFlt>::TIter iter=RNScores.BegI();!iter.IsEnd();iter++) {
      const TInt NId = iter.GetKey();
      if (RNNIds.Len()>=RNNumber) break;

      if (RNNIds.Len()==0) {
         RNNIds.Add(NId);
         printf("%d representative node selected with score %f, occurrence %f, uniqueness %f\n", NId(), RNScores.GetDat(NId)(),occurrenceTimes.GetDat(NId)(), uniqueness.GetDat(NId)());
      }
      else {
         bool selected = true;
         for (TVec<TInt>::TIter RNI=RNNIds.BegI();RNI!=RNNIds.EndI();RNI++) {
            const TInt RNNId = *RNI;
            TFlt entropy = crossEntropy(RNNId,NId);
            if (entropy < EntropyThreshold) {
               selected = false;
               break; 
            }
         }
         
         if (selected) {
           RNNIds.Add(NId);
           printf("%d representative node selected with score %f, occurrence %f, uniqueness %f\n", NId(), RNScores.GetDat(NId)(),occurrenceTimes.GetDat(NId)(), uniqueness.GetDat(NId)());
         }
      }
   }

   RNNumber = RNNIds.Len();
}

TFlt RNCascade::crossEntropy(const TInt &NId1, const TInt &NId2) const {
   TFlt sum = 0.0;
   for (THash<TInt, TCascade>::TIter iter=CascH.BegI();!iter.IsEnd();iter++) {
      TFlt p,q;
      TCascade &cascade = iter.GetDat();

      if (cascade.IsNode(NId1)) p = 0.95;
      else p = 0.05;
      if (cascade.IsNode(NId2)) q = 0.95;
      else q = 0.05;
     
      sum += -p * TMath::Log(q);
   }
   return sum/CascH.Len();
}

bool RNCascade::isRNCategory(const int& RNNId, TCascade& Cascade) {
   if (Cascade.IsNode(RNNId)) {
      /*for (int i=0;i<RNNIds.Len();i++) {
         int NId = RNNIds[i];
         if (NId==RNNId || !Cascade.IsNode(NId)) continue;
         if (Cascade.GetTm(NId) < Cascade.GetTm(RNNId)) return false;
      }*/

      double RNInfectedTime = Cascade.GetTm(RNNId) - Cascade.GetMinTm();
      double cascadeTimeWindow = Cascade.GetMaxTm() - Cascade.GetMinTm();
      if (RNInfectedTime < cascadeTimeWindow*0.1) return true;
   }
   return false;
}
