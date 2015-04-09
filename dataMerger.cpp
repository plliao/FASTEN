#include "stdafx.h"
#include "RNCascade.h"

int main(int argc, char* argv[]) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nStochastic network inference. build: %s, %s. Time: %s", __TIME__, __DATE__, TExeTm::GetCurTm()));
  TExeTm ExeTm;
  Try

  const TStr InFNm  = Env.GetIfArgPrefixStr("-i:", "example-cascades.txt", "Input cascades");
  const TStr GroundTruthFNm = Env.GetIfArgPrefixStr("-n:", "example-network.txt", "Input ground-truth network");
  const TStr OutFNm  = Env.GetIfArgPrefixStr("-o:", "network", "Output file name(s) prefix");

  const TModel Model = (TModel)Env.GetIfArgPrefixInt("-m:", 0, "0:exponential, 1:power law, 2:rayleigh, 3:weibull");


  RNCascade rNCascade;
  printf("\nLoading input cascades: %s\n", InFNm.CStr());

  rNCascade.SetModel(Model);

  TStrV InFNms, GTFNms;
  InFNm.SplitOnAllCh(':',InFNms);
  GroundTruthFNm.SplitOnAllCh(':',GTFNms);

  // load cascades from file
  TFIn FIn(InFNms[0]);
  rNCascade.LoadCascadesTxt(FIn);
  for (int i=1;i<InFNms.Len();i++) {
     printf("%s\n", InFNms[i].CStr());
     TFIn FIn(InFNms[i]);
     rNCascade.LoadMultipleCascadesTxt(FIn);
  }

  // load ground truth network
  TFIn FGIn(GTFNms[0]);
  rNCascade.LoadGroundTruthTxt(FGIn);
  for (int i=1;i<GTFNms.Len();i++) {
     printf("%s\n", GTFNms[i].CStr());
     TFIn FGIn(GTFNms[i]);
     rNCascade.LoadMultipleGroundTruthTxt(FGIn);
  }

  // Save inferred network in a file
  rNCascade.SaveGroundTruth(TStr::Fmt("%s.txt", OutFNm.CStr()));
  rNCascade.SaveCascades(TStr::Fmt("%s-cascade.txt",OutFNm.CStr())); 
 
  Catch
  printf("\nrun time: %s (%s)\n", ExeTm.GetTmStr(), TSecTm::GetCurTm().GetTmStr().CStr());
  return 0;
}
