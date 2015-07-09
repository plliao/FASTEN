#include <algorithm>
#include "stdafx.h"
#include <cascdynetinf.h>
#include <Evaluator.h>

int main(int argc, char* argv[]) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nStochastic network inference. build: %s, %s. Time: %s", __TIME__, __DATE__, TExeTm::GetCurTm()));
  TExeTm ExeTm;
  Try

  const TStr InFNm  = Env.GetIfArgPrefixStr("-i:", "example-inferred", "Input inferred networks");
  const TStr GroundTruthFNm = Env.GetIfArgPrefixStr("-n:", "example-network.txt", "Input ground-truth network");
  const TStr OutFNm  = Env.GetIfArgPrefixStr("-o:", "network", "Output file name(s) prefix");
  const TStr modelNm  = Env.GetIfArgPrefixStr("-m:", "InfoPath", "Input model name(s)");
  const TInt k = Env.GetIfArgPrefixInt("-k:", 3, "topic size");

  TStrV InFNms, modelNms, GroundTruthNms;
  InFNm.SplitOnAllCh(':',InFNms);
  modelNm.SplitOnAllCh(':',modelNms);
  GroundTruthFNm.SplitOnAllCh('-',GroundTruthNms);

  TVec<Evaluator> evaluators(k);
  for (TInt i=0; i<k; i++) {
     evaluators.Add(Evaluator());
     TStr SubGroundTruthFNm = GroundTruthNms[0] + TStr::Fmt("-%d-",i+1) + GroundTruthNms[1];
     printf("\nLoading input ground truth %d: %s\n", i(), SubGroundTruthFNm.CStr());
     // load ground truth network
     TFIn FGIn(SubGroundTruthFNm);
     evaluators[i()].LoadGroundTruth(FGIn);
  
     printf("\nLoading input inferred networks: %s\n", InFNm.CStr());
     for (int r=0;r<InFNms.Len();r++) {
        printf("\nLoading %s\n", InFNms[r].CStr());
        for (TInt j=0; j<k; j++) {
           TStr SubInFNm = InFNms[r] + TStr("_") + j.GetStr() + ".txt";
           TFIn FIn(SubInFNm);
           evaluators[i()].LoadInferredNetwork(FIn, modelNms[r] + TStr("_") + j.GetStr());
           printf("%s\n",SubInFNm.CStr());
        }
        printf("\nLoading %s done\n", InFNms[r].CStr());
     }

     TVec<TFlt> steps1 = evaluators[i()].GetSteps(0);

     evaluators[i()].EvaluatePRC(steps1.Last(),false);
     evaluators[i()].EvaluateAUC(steps1.Last());
     evaluators[i()].EvaluateMSE(steps1.Last());
  }

  printf("\n");

  TVec<TFlt> steps1 = evaluators[0].GetSteps(0);
  TFlt time = steps1.Last();
  TInt size = 1;
  TIntV indexs(k);
  TStrV permutation;
  for (TInt i=0; i<k; i++) { 
     size = size * (i+1);
     indexs[i] = i;
  }
  THash<TIntPr, TFlt> PRC_AUCTable, MSETable;

  for (int r=0;r<InFNms.Len();r++) {
     printf("\n");
     TFlt maxAUC = -1.0;
     TInt maxIndex = -1, index = 0;
     do {
       TStr permutationString = modelNms[r];
       TFlt PRC_AUC = 0.0, MSE = 0.0;
       for (TInt i=0; i<k; i++) {
          PRC_AUC += evaluators[i()].PRC_AUC[r*k + indexs[i()]].GetDat(time);
          MSE += evaluators[i()].MSE[r*k + indexs[i()]].GetDat(time);
          permutationString += TStr::Fmt(" %d,",indexs[i()]);
       } 
       if (PRC_AUC > maxAUC) {
          maxAUC = PRC_AUC;
          maxIndex = index;
       }
       permutationString += TStr::Fmt(" PRC_AUC:%f, MSE:%f\n", PRC_AUC/k, MSE/k);
       printf(permutationString.CStr());
       permutation.Add(permutationString);
       index++;
     } while ( std::next_permutation(indexs.BegI(),indexs.EndI()) );

     printf("%s Best AUC: %s", modelNms[r].CStr(), permutation[maxIndex].CStr());
     permutation.Clr();
  }
 
  Catch
  printf("\nrun time: %s (%s)\n", ExeTm.GetTmStr(), TSecTm::GetCurTm().GetTmStr().CStr());
  return 0;
}
