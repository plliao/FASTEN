#include "stdafx.h"
#include <FASTENModel.h>

int main(int argc, char* argv[]) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nStochastic network inference. build: %s, %s. Time: %s", __TIME__, __DATE__, TExeTm::GetCurTm()));
  TExeTm ExeTm;
  Try

  bool verbose = true;

  const TStr InFNm  = Env.GetIfArgPrefixStr("-i:", "example-cascades.txt", "Input cascades");
  const TStr GroundTruthFNm = Env.GetIfArgPrefixStr("-n:", "example-network.txt", "Input ground-truth network");
  const TStr OutFNm  = Env.GetIfArgPrefixStr("-o:", "network", "Output file name(s) prefix");

  const TModel Model = (TModel)Env.GetIfArgPrefixInt("-m:", 0, "0:exponential, 1:power law, 2:rayleigh, 3:weibull");
  const double Delta = Env.GetIfArgPrefixFlt("-d:", 1.0, "Delta for power-law (default:1)\n"); // delta for power law
  
  const int latentVariableSize  = Env.GetIfArgPrefixInt("-K:", 10, "Latent variable size");

  const TRunningMode RunningMode = (TRunningMode)Env.GetIfArgPrefixInt("-rm:", 0,"Running mode\n0:Time step, 1:Infections step, 2:Cascade step, 3:Single time point\n");
  const double TimeStep = Env.GetIfArgPrefixFlt("-ts:", 10.0, "Minimum time step size for -rm:0 (default:10.0)\n");
  const int NumberInfections = Env.GetIfArgPrefixInt("-is:", -1, "Number of infections for -rm:1  (default:-1)\n");

  const TStr MinTimeStr = Env.GetIfArgPrefixStr("-it:", "-1", "First time (default:-1)\n");
  const TStr MaxTimeStr = Env.GetIfArgPrefixStr("-tt:", "-1", "Last time (default:-1)\n");
  const int InputTimeFormat = Env.GetIfArgPrefixInt("-itf:",0, "First/last time format\n0:no time units,1:a date to be parsed\n");
  const int CascadesTimeFormat = Env.GetIfArgPrefixInt("-ctf:", 0, "Cascades time format\n0:no time units,1:second,2:minute,3:hour,4:6 hours,5:12 hours, 6:day\n");

  double Window = Env.GetIfArgPrefixFlt("-h:", -1, "Time window per cascade (if any), -1 means the window is set by  means last infection time (default:-1)\n");
  double observedWindow = Env.GetIfArgPrefixFlt("-w:", 10.0, "Observed time window (default 10.0)\n");

  const TStr NodeIdx = Env.GetIfArgPrefixStr("-ni:", "-1", "Node indeces to estimate incoming tx rates (-1:all nodes, -X:random -X-node subset)");

  const TSampling TSam = (TSampling)Env.GetIfArgPrefixInt("-t:", 0, "Sampling method\n0:UNIF_SAMPLING, 1:WIN_SAMPLING, 2:EXP_SAMPLING, 3:WIN_EXP_SAMPLING, 4:RAY_SAMPLING");
  const int Iters  = Env.GetIfArgPrefixInt("-e:", 1000, "Number of iterations per time step");
  const int EMIters  = Env.GetIfArgPrefixInt("-em:", 5, "Number of iterations of expectation maximization");
  const int BatchLen = Env.GetIfArgPrefixInt("-bl:", 1, "Number of cascades for each batch, -t:2 & -t:4 (default:1000)");
  const TStr ParamSampling = Env.GetIfArgPrefixStr("-sd:", "0.1", "Params for -t:1,2 & -t:4,5 (default:0.1)\n");

  const double lr = Env.GetIfArgPrefixFlt("-g:", 0.001, "Alpha for gradient descend (default:0.01)\n");
  const double Aging = Env.GetIfArgPrefixFlt("-a:", 1.0, "Aging factor for non-used edges (default:1.0)\n");
  const TRegularizer Regularizer = (TRegularizer)Env.GetIfArgPrefixInt("-r:", 0, "Regularizer\n0:no, 1:l2");
  const double Mu = Env.GetIfArgPrefixFlt("-mu:", 0.01, "Mu for regularizer (default:0.01)\n");
  const double decayRatio = Env.GetIfArgPrefixFlt("-df:", 3.0, "Damping factor (default:3.0)\n");

  const double Tol = Env.GetIfArgPrefixFlt("-tl:", 0.0005, "Tolerance (default:0.01)\n");
  const double MinAlpha = Env.GetIfArgPrefixFlt("-la:", 0.05, "Min alpha (default:0.05)\n");
  const double MaxAlpha = Env.GetIfArgPrefixFlt("-ua:", 100, "Maximum alpha (default:100)\n");
  const double InitAlpha = Env.GetIfArgPrefixFlt("-ia:", 0.01, "Initial alpha (default:0.01)\n");

  FASTENModel fasten;
  printf("\nLoading input cascades: %s\n", InFNm.CStr());

  fasten.SetModel(Model);
  fasten.SetDelta(Delta);
  fasten.SetSampling(TSam);
  fasten.SetMaxIterNm(Iters);
  fasten.SetMaxEMIterNm(EMIters);
  fasten.SetBatchSize(BatchLen);
  fasten.SetLearningRate(lr);
  fasten.SetParamSampling(ParamSampling);

  fasten.SetLatentVariableSize(latentVariableSize);
  fasten.SetTolerance(Tol);
  fasten.SetMaxAlpha(MaxAlpha);
  fasten.SetMinAlpha(MinAlpha);
  fasten.SetInitAlpha(InitAlpha);
  fasten.SetRegularizer(Regularizer);
  fasten.SetMu(Mu);
  fasten.SetWindow(Window);
  fasten.SetObservedWindow(observedWindow);
  fasten.SetAging(Aging);
  fasten.SetDecayRatio(decayRatio);

  // load cascades from file
  fasten.LoadCascadesTxt(InFNm);
  
  printf("cascades:%d\nRunning Stochastic Network Inference..\n", fasten.GetCascs());

  double MaxTime = TFlt::Mn;
  double MinTime = TFlt::Mx;

  if (MaxTimeStr.EqI("-1")) {
    // find maximum time across cascades
    for (int i=0; i<fasten.GetCascs(); i++) {
      if (fasten.CascH[i].GetMaxTm() > MaxTime) {
        MaxTime = fasten.CascH[i].GetMaxTm();
      }
    }
  } else {
    if (InputTimeFormat==0) { MaxTime = MaxTimeStr.GetFlt(); }
    else {
      MaxTime = (double)TSecTm::GetDtTmFromStr(MaxTimeStr).GetAbsSecs();
    }
  }

  if (MinTimeStr.EqI("-1")) {
    // find minimum time across cascades
    MinTime = TFlt::Mx;
    for (int i=0; i<fasten.GetCascs(); i++) {
      if (fasten.CascH[i].GetMinTm() < MinTime && fasten.CascH[i].GetMinTm()!=0) {
        MinTime = fasten.CascH[i].GetMinTm();
      }
    }
  } else {
    if (InputTimeFormat==0) { MinTime = MinTimeStr.GetFlt(); }
    else {
      MinTime = (double)TSecTm::GetDtTmFromStr(MinTimeStr).GetAbsSecs();
    }
  }

  if (MinTime<1.0) { MinTime = 0.0; }

  if (InputTimeFormat==1) {
    switch(CascadesTimeFormat) {
        case 0: case 1: break;
        case 2: MinTime /= 60.0; MaxTime /= 60.0; break;
        case 3: MinTime /= 3600.0; MaxTime /= 3600.0; break;
        case 4: MinTime /= (6.0*3600.0); MaxTime /= (6.0*3600.0); break;
        case 5: MinTime /= (12.0*3600.0); MaxTime /= (12.0*3600.0); break;
        case 6: MinTime /= (24.0*3600.0); MaxTime /= (24.0*3600.0); break;
        default: FailR("Bad -s: parameter.");
    }
  }

  TFltV Steps;
  int num_infections = 0;
  TFltV InfectionsV;

  // we always add 0.0 as starting point
  Steps.Add(MinTime);

  printf("Starting time: %f\n", MinTime);

  // compute number of time points to compute estimation for each running mode
  switch (RunningMode) {
      case TIME_STEP:
        for (float t = MinTime+TimeStep; t<=MaxTime; t += TimeStep) { Steps.Add(t); if (verbose) { printf("Time: %f\n", Steps[Steps.Len()-1].Val); } }
        break;
      case INFECTION_STEP:
        // copy infections
        if (verbose) { printf("Generating infections vector...\n"); }
        for (int i=0; i<fasten.GetCascs(); i++) {
          for (int j=0; j<fasten.CascH[i].Len() && fasten.CascH[i].NIdHitH[j].Tm < MaxTime; j++) {
            InfectionsV.Add(fasten.CascH[i].NIdHitH[j].Tm);
          }
        }

        // sort infections
        if (verbose) { printf("Infections vector generated (%d infections)!\nSorting infections...\n", InfectionsV.Len()); }
        InfectionsV.Sort(true);
        if (verbose) { printf("Infections sorted!\n"); }

        // generate time steps
        for (int i=0; i<InfectionsV.Len(); i++, num_infections++) {
          if (num_infections==NumberInfections) {
            if (InfectionsV[i]!=Steps[Steps.Len()-1]) { Steps.Add(InfectionsV[i]); }
            num_infections = 0;
            if (verbose) { printf("Time: %f\n", Steps[Steps.Len()-1].Val); }
          }
        }
        break;

      case CASCADE_STEP:
        for (int i=0; i<fasten.GetCascs(); i++) {
          if (fasten.CascH[i].GetMaxTm()<MinTime+MaxTime) { Steps.Add(fasten.CascH[i].GetMaxTm()); }
        }

        Steps.Sort();

        if (verbose) { for (int i=0; i<Steps.Len(); i++) { printf("Time: %f\n", Steps[i].Val); } }
        break;

      case SINGLE_STEP:
        Steps.Add(MinTime+MaxTime);
        if (verbose) { printf("Time: %f\n", Steps[0].Val); }
        break;

      default:
      FailR("Bad -rm: parameter.");

  }

  // save time steps
  TFOut FOutTimeSteps(TStr::Fmt("%s-time-steps.txt", OutFNm.CStr()));
  for (int i=0; i<Steps.Len(); i++) { FOutTimeSteps.PutStr(TStr::Fmt("%f\n", Steps[i].Val)); }

  fasten.Init();
  fasten.Infer(Steps, OutFNm);
  fasten.SaveInferred(TStr::Fmt("%s.txt", OutFNm.CStr()));
  fasten.SavePriorTopicProbability(TStr::Fmt("%s_PriorTopicProbability.txt", OutFNm.CStr()));
  
  Catch
  printf("\nrun time: %s (%s)\n", ExeTm.GetTmStr(), TSecTm::GetCurTm().GetTmStr().CStr());
  return 0;
}
