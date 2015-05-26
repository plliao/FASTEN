#ifndef INFOPATHFILEIO_H
#define INFOPATHFILEIO_H

#include <cascdynetinf.h>

struct NodeInfo {
   THash<TInt, TNodeInfo> NodeNmH;
   TStrIntH DomainsIdH;
   TModel Model;
};

struct EdgeInfo {
   TFlt MaxAlpha, MinAlpha;
};

class InfoPathFileIO {
   public:
      static void LoadCascadesTxt(TSIn& SIn, THash<TInt, TCascade>& CascH, NodeInfo &nodeInfo, bool verbose=false);
      static void LoadNetworkTxt(TSIn& SIn, TStrFltFltHNEDNet& Network, NodeInfo &nodeInfo, bool verbose=false);

      static void AddCascadesTxt(TSIn& SIn, THash<TInt, TCascade>& CascH, NodeInfo &nodeInfo, bool verbose=false);
      static void AddNetworkTxt(TSIn& SIn, TStrFltFltHNEDNet& Network, NodeInfo &nodeInfo, bool verbose=false);

      static void SaveNetwork(const TStr& OutFNm, TStrFltFltHNEDNet& Network, NodeInfo &nodeInfo, EdgeInfo &edgeInfo, const TIntV& NIdV=TIntV());
      static void SaveCascades(const TStr& OutFNm, THash<TInt, TCascade>& CascH, NodeInfo &nodeInfo); 

      static void GenerateInferredNetwork(TStrFltFltHNEDNet& Network, THash<TInt,TStrFltFltHNEDNet>& MultipleNetworks);
   private:
      static void AddCasc(const TStr& CascStr, THash<TInt, TCascade>& CascH, NodeInfo &nodeInfo, int CId=-1);
      static void AddNodeNm(const int& NId, const TNodeInfo& Info, NodeInfo &nodeInfo);
      static void AddDomainNm(const TStr& Domain, NodeInfo &nodeInfo, const int& DomainId=-1);

      static bool IsCascade(int c, THash<TInt, TCascade>& CascH) { return CascH.IsKey(c); } 
      static bool IsNodeNm(const int& NId, NodeInfo &nodeInfo) { return nodeInfo.NodeNmH.IsKey(NId); }
      static bool IsDomainNm(const TStr& Domain, NodeInfo &nodeInfo) { return nodeInfo.DomainsIdH.IsKey(Domain); }

      static void LoadNodes(TSIn& SIn, NodeInfo &nodeInfo, bool verbose=false);
      static void LoadAndAddNodes(TSIn& SIn, TStrFltFltHNEDNet& Network, NodeInfo &nodeInfo, bool verbose=false);

};

#endif
