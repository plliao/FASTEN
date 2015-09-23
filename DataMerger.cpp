#include "stdafx.h"
#include <InfoPathFileIO.h>
#include <cascdynetinf.h>

int main(int argc, char* argv[]) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nStochastic network inference. build: %s, %s. Time: %s", __TIME__, __DATE__, TExeTm::GetCurTm()));
  TExeTm ExeTm;
  Try

  const TStr InFNm  = Env.GetIfArgPrefixStr("-i:", "example-cascades.txt", "Input cascades");
  const TStr GroundTruthFNm = Env.GetIfArgPrefixStr("-n:", "example-network.txt", "Input ground-truth network");
  const TStr OutFNm  = Env.GetIfArgPrefixStr("-o:", "network", "Output file name(s) prefix");

  const TModel Model = (TModel)Env.GetIfArgPrefixInt("-m:", 0, "0:exponential, 1:power law, 2:rayleigh, 3:weibull");
  
  const double MinAlpha = Env.GetIfArgPrefixFlt("-la:", 0.05, "Min alpha (default:0.05)\n");
  const double MaxAlpha = Env.GetIfArgPrefixFlt("-ua:", 100, "Maximum alpha (default:100)\n");

  TStrFltFltHNEDNet GroundTruth;
  THash<TInt, TCascade> CascH;
  NodeInfo nodeInfo;
  nodeInfo.Model = Model;
  EdgeInfo edgeInfo = {MaxAlpha, MinAlpha};

  TStrV InFNms, GTFNms;
  InFNm.SplitOnAllCh(':',InFNms);
  GroundTruthFNm.SplitOnAllCh(':',GTFNms);

  // load cascades from file
  printf("\nLoading input cascades: %s\n", InFNm.CStr());
  TFIn FIn(InFNms[0]);
  InfoPathFileIO::LoadCascadesTxt(FIn, CascH, nodeInfo);
 
  for (int i=1;i<InFNms.Len();i++) {
     printf("%s\n", InFNms[i].CStr());
     TFIn FIn(InFNms[i]);
     InfoPathFileIO::AddCascadesTxt(FIn, CascH, nodeInfo);
  }

  printf("\nLoading input networks: %s\n", GroundTruthFNm.CStr());
  // load ground truth network
  TFIn FGIn(GTFNms[0]);
  InfoPathFileIO::LoadNetworkTxt(FGIn, GroundTruth, nodeInfo);
  
  for (int i=1;i<GTFNms.Len();i++) {
     printf("%s\n", GTFNms[i].CStr());
     TFIn FGIn(GTFNms[i]);
     InfoPathFileIO::AddNetworkTxt(FGIn, GroundTruth, nodeInfo);
  }

  // Save inferred network in a file
  InfoPathFileIO::SaveNetwork(TStr::Fmt("%s-network.txt", OutFNm.CStr()), GroundTruth, nodeInfo, edgeInfo);
  InfoPathFileIO::SaveCascades(TStr::Fmt("%s-cascades.txt",OutFNm.CStr()), CascH, nodeInfo); 
 
  Catch
  printf("\nrun time: %s (%s)\n", ExeTm.GetTmStr(), TSecTm::GetCurTm().GetTmStr().CStr());
  return 0;
}
