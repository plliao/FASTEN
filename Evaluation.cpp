#include "stdafx.h"
#include <cascdynetinf.h>
#include <Evaluator.h>

int main(int argc, char* argv[]) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nStochastic network inference. build: %s, %s. Time: %s", __TIME__, __DATE__, TExeTm::GetCurTm()));
  TExeTm ExeTm;
  Try

  const TStr InFNm  = Env.GetIfArgPrefixStr("-i:", "example-inferred-network.txt", "Input inferred networks");
  const TStr GroundTruthFNm = Env.GetIfArgPrefixStr("-n:", "example-network.txt", "Input ground-truth network");
  const TStr OutFNm  = Env.GetIfArgPrefixStr("-o:", "network", "Output file name(s) prefix");
  const TStr modelNm  = Env.GetIfArgPrefixStr("-m:", "InfoPath", "Input model name(s)");

  const double MinAlpha = Env.GetIfArgPrefixFlt("-la:", 0.0, "Min alpha (default:0.05)\n");
  const double MaxAlpha = Env.GetIfArgPrefixFlt("-ua:", 100, "Maximum alpha (default:100)\n");
  
  const double PRCPointNm = Env.GetIfArgPrefixFlt("-p:", 10000, "PRC points number (default:10000)\n");

  TStrV InFNms, modelNms;
  InFNm.SplitOnAllCh(':',InFNms);
  modelNm.SplitOnAllCh(':',modelNms);

  Evaluator evaluator;
  printf("\nLoading input ground truth: %s\n", GroundTruthFNm.CStr());
  // load ground truth network
  TFIn FGIn(GroundTruthFNm);
  evaluator.LoadGroundTruth(FGIn);
  
  printf("\nLoading input inferred networks: %s\n", InFNm.CStr());
  for (int i=0;i<InFNms.Len();i++) {
     printf("\nLoading %s\n", InFNms[i].CStr());
     TFIn FIn(InFNms[i]);
     evaluator.LoadInferredNetwork(FIn, modelNms[i]);
     printf("\nLoading %s done\n", InFNms[i].CStr());
  }

  TVec<TFlt> steps1 = evaluator.GetSteps(0);

  evaluator.EvaluatePRC(MinAlpha, MaxAlpha, steps1.Last(),PRCPointNm);
  evaluator.EvaluateAUC(steps1.Last());
  evaluator.EvaluateMSE(steps1.Last());
  evaluator.PlotPRC(OutFNm);
  evaluator.PlotMSE(OutFNm);
 
  Catch
  printf("\nrun time: %s (%s)\n", ExeTm.GetTmStr(), TSecTm::GetCurTm().GetTmStr().CStr());
  return 0;
}
