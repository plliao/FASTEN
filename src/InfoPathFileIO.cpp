#include <InfoPathFileIO.h>

void InfoPathFileIO::LoadNodes(TSIn& SIn, NodeInfo &nodeInfo, bool verbose) {
  TStr Line;
  while (!SIn.Eof()) {
    SIn.GetNextLn(Line);
    if (Line=="") { break; }

    TStrV NIdV; Line.SplitOnAllCh(',', NIdV);
    const int NId = NIdV[0].GetInt();
    const TStr domainName = NIdV[1];

    if (!IsNodeNm(NId, nodeInfo)) AddNodeNm(NId, TNodeInfo(domainName, 0), nodeInfo);
    else IAssert(domainName == nodeInfo.NodeNmH.GetDat(NId).Name); 

    if (!IsDomainNm(domainName, nodeInfo)) AddDomainNm(domainName, nodeInfo, NId);
    else IAssert(NId == nodeInfo.DomainsIdH.GetDat(domainName));
    
  }
  if (verbose) printf("All nodes read!\n");

}

void InfoPathFileIO::LoadAndAddNodes(TSIn& SIn, TStrFltFltHNEDNet& Network, NodeInfo &nodeInfo, bool verbose) {
  TStr Line;
  while (!SIn.Eof()) {
    SIn.GetNextLn(Line);
    if (Line=="") { break; }

    TStrV NIdV; Line.SplitOnAllCh(',', NIdV);
    const int NId = NIdV[0].GetInt();
    const TStr domainName = NIdV[1];

    if (!IsNodeNm(NId, nodeInfo)) {
       AddNodeNm(NId, TNodeInfo(domainName, 0), nodeInfo);
       Network.AddNode(NId, domainName);
    }
    else IAssert(domainName == nodeInfo.NodeNmH.GetDat(NId).Name); 

    if (!IsDomainNm(domainName, nodeInfo)) AddDomainNm(domainName, nodeInfo, NId);
    else IAssert(NId == nodeInfo.DomainsIdH.GetDat(domainName));
    
  }
  if (verbose) printf("All nodes read!\n");

}

void InfoPathFileIO::LoadCascadesTxt(TSIn& SIn, THash<TInt, TCascade>& CascH, NodeInfo &nodeInfo, bool verbose) {
  LoadNodes(SIn, nodeInfo, verbose);
  TStr Line;
  while (!SIn.Eof()) { SIn.GetNextLn(Line); AddCasc(Line, CascH, nodeInfo); }
  if (verbose) printf("All cascades read!\n");
}

void InfoPathFileIO::LoadNetworkTxt(TSIn& SIn, TStrFltFltHNEDNet& Network, NodeInfo &nodeInfo, bool verbose) {
  TStr Line;

  Network.Clr(); // clear network (if any)
   
  LoadNodes(SIn, nodeInfo, verbose);
  for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
    Network.AddNode(NI.GetKey(), NI.GetDat().Name);
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

    Network.AddEdge(FieldsV[0].GetInt(), FieldsV[1].GetInt(), Alphas);

    if (verbose) {
      printf("Edge %d -> %d: ", FieldsV[0].GetInt(), FieldsV[1].GetInt());
      TFltFltH &AlphasE = Network.GetEDat(FieldsV[0].GetInt(), FieldsV[1].GetInt());
      for (int i=0; i<AlphasE.Len(); i+=2) { printf("(%f, %f)", AlphasE.GetKey(i).Val, AlphasE[i].Val); }
      printf("\n");
    }
  }

  if (verbose) printf("network nodes:%d edges:%d\n", Network.GetNodes(), Network.GetEdges());
} 

void InfoPathFileIO::AddCascadesTxt(TSIn& SIn, THash<TInt, TCascade>& CascH, NodeInfo &nodeInfo, bool verbose) {
   LoadNodes(SIn, nodeInfo, verbose);
   TStr Line;
   while (!SIn.Eof()) { SIn.GetNextLn(Line); AddCasc(Line, CascH, nodeInfo, CascH.Len()); }
   if (verbose) printf("All cascades read!\n");
}

void InfoPathFileIO::AddNetworkTxt(TSIn& SIn, TStrFltFltHNEDNet& Network, NodeInfo &nodeInfo, bool verbose) {
   LoadAndAddNodes(SIn, Network, nodeInfo, verbose);
  
   TStr Line;
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

   if (verbose) printf("network nodes:%d edges:%d\n", Network.GetNodes(), Network.GetEdges());
}

void InfoPathFileIO::SaveNetwork(const TStr& OutFNm, TStrFltFltHNEDNet& Network, NodeInfo &nodeInfo, EdgeInfo &edgeInfo, const TIntV& NIdV) {
  TFOut FOut(OutFNm);

  // write nodes to file
  for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
    if (NIdV.Len() > 0 && !NIdV.IsIn(NI.GetKey())) { continue; }

    FOut.PutStr(TStr::Fmt("%d,%s\r\n", NI.GetKey().Val, NI.GetDat().Name.CStr()));
  }

  FOut.PutStr("\r\n");

  // write edges to file (not allowing self loops in the network)
  for (TStrFltFltHNEDNet::TEdgeI EI = Network.BegEI(); EI < Network.EndEI(); EI++) {
    if (NIdV.Len() > 0 && (!NIdV.IsIn(EI.GetSrcNId()) || !NIdV.IsIn(EI.GetDstNId()))) { continue; }
    if (!nodeInfo.NodeNmH.IsKey(EI.GetSrcNId()) || !nodeInfo.NodeNmH.IsKey(EI.GetDstNId())) { continue; }

    // not allowing self loops in the Kronecker network
    if (EI.GetSrcNId() != EI.GetDstNId()) {
      if (EI().Len() > 0) {
        TStr Line; bool IsEdge = false;
        for (int i=0; i<EI().Len(); i++) {
          if (EI()[i]> edgeInfo.MinAlpha) {
            Line += TStr::Fmt(",%f,%f", EI().GetKey(i).Val, (EI()[i] > edgeInfo.MaxAlpha? edgeInfo.MaxAlpha.Val : EI()[i].Val) );
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
}

void InfoPathFileIO::SaveCascades(const TStr& OutFNm, THash<TInt, TCascade>& CascH, NodeInfo &nodeInfo) {
  TFOut FOut(OutFNm);

  // write nodes to file
  for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
    FOut.PutStr(TStr::Fmt("%d,%s\r\n", NI.GetKey().Val, NI.GetDat().Name.CStr()));
  }

  FOut.PutStr("\r\n");

  // write cascades to file
  for (THash<TInt, TCascade>::TIter CI = CascH.BegI(); CI < CascH.EndI(); CI++) {
    TCascade &C = CI.GetDat();
    int j = 0;
    for (THash<TInt, THitInfo>::TIter NI = C.NIdHitH.BegI(); NI < C.NIdHitH.EndI(); NI++) {
      if (!nodeInfo.NodeNmH.IsKey(NI.GetDat().NId)) { continue; }
      if (j > 0) { FOut.PutStr(TStr::Fmt(",%d,%f", NI.GetDat().NId.Val, NI.GetDat().Tm.Val)); }
      else { FOut.PutStr(TStr::Fmt("%d;%d,%f", CI.GetKey().Val, NI.GetDat().NId.Val, NI.GetDat().Tm.Val)); }
      j++;
    }

    if (j >= 1)
      FOut.PutStr(TStr::Fmt("\r\n"));
  }
}

void InfoPathFileIO::AddCasc(const TStr& CascStr, THash<TInt, TCascade>& CascH, NodeInfo &nodeInfo, int CId) {
  // support cascade id if any
  TStrV FieldsV; CascStr.SplitOnAllCh(';', FieldsV);
  if (FieldsV.Len()==2) { 
     if (CId==-1) CId = FieldsV[0].GetInt(); 
  }

  // read nodes
  TStrV NIdV; FieldsV[FieldsV.Len()-1].SplitOnAllCh(',', NIdV);
  TCascade C(CId, nodeInfo.Model);
  for (int i = 0; i < NIdV.Len(); i+=2) {
    int NId = NIdV[i].GetInt();
    double Tm = NIdV[i+1].GetFlt();
    nodeInfo.NodeNmH.GetDat(NId).Vol += 1;
    C.Add(NId, Tm);
  }
  C.Sort();
  CascH.AddDat(C.CId) = C;
}

void InfoPathFileIO::AddNodeNm(const int& NId, const TNodeInfo& Info, NodeInfo &nodeInfo) {
   nodeInfo.NodeNmH.AddDat(NId, Info);  
}

void InfoPathFileIO::AddDomainNm(const TStr& Domain, NodeInfo &nodeInfo, const int& DomainId) {
   nodeInfo.DomainsIdH.AddDat(Domain) = TInt(DomainId==-1? nodeInfo.DomainsIdH.Len() : DomainId); 
}

void InfoPathFileIO::GenerateInferredNetwork(TStrFltFltHNEDNet& Network, THash<TInt,TStrFltFltHNEDNet>& MultipleNetworks) {
   for (THash<TInt,TStrFltFltHNEDNet>::TIter NI = MultipleNetworks.BegI(); !NI.IsEnd(); NI++) {
      printf("GenerateInferredNetwork: %d\n",NI.GetKey()());
      for (TStrFltFltHNEDNet::TEdgeI EI = NI.GetDat().BegEI(); EI < NI.GetDat().EndEI(); EI++) {
         int srcNId = EI.GetSrcNId(), dstNId = EI.GetDstNId();
         if (!Network.IsEdge(srcNId,dstNId)) Network.AddEdge(srcNId,dstNId,EI.GetDat());
         else {
            TFltFltH& alpha = Network.GetEDat(srcNId,dstNId);
            for (TFltFltH::TIter AI = EI.GetDat().BegI(); !AI.IsEnd(); AI++) {
               TFlt time = AI.GetKey();
               if(!alpha.IsKey(time)) alpha.AddDat(time,AI.GetDat());
               else {
                  if(alpha.GetDat(time) < AI.GetDat()) alpha.GetDat(time) = AI.GetDat();
               }
            }
         }
      }
   }
}
