#include "stdafx.h"
#include <DecayCascadesModel.h>
#include <InfoPathFileIO.h>

int main(int argc, char* argv[]) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nGenerate different synthetic networks & cascades. build: %s, %s. Time: %s", __TIME__, __DATE__, TExeTm::GetCurTm()));
  TExeTm ExeTm;
  Try
  TInt::Rnd.PutSeed(0);
  TFlt::Rnd.PutSeed(0);

  const int TNetwork = Env.GetIfArgPrefixInt("-t:", 0, "Network to generate\n\
  	  0:kronecker, 1:forest fire, 2:given in txt, gen only cascades (default:0)\n"); // network (kronecker, forest-fire, struct&rates given in txt)
  const TStr NetworkParams = Env.GetIfArgPrefixStr("-g:", TStr("0.9 0.5; 0.5 0.9"), "Parameters for the network (default:0.9 0.5; 0.5 0.9)\n"); // network params for kronecker/forest-fire
  const TStr GroundTruthFileName = Env.GetIfArgPrefixStr("-in:", TStr("input"), "Name of the input network (default:input)\n");  // input groundtruth, if any

  // edge types (constant, linear, rayleigh, exponential, slab, random)
  const TStr VaryingType = Env.GetIfArgPrefixStr("-vt:", "1;0;0;0;0;0;0;0", "Varying trends percentages for tx rates\nconstant;linear;exponential;rayleigh;slab;square;chainsaw;random (default:1;0;0;0;0;0)");
  //const double ResolutionPerEdge = Env.GetIfArgPrefixFlt("-nvt:", 1.0, "Time resolution for tx rate evolution (default: 1.0)");
  const TStr VaryingTypeParameter = Env.GetIfArgPrefixStr("-dvt:", "20;1", "Period;decay for rayleigh/exponential/slab/square/chainsaw (default: 20;1)");

  // nodes, edges
  const int NNodes = Env.GetIfArgPrefixInt("-n:", 512, "Number of nodes (default:512)\n");
  const int NEdges = Env.GetIfArgPrefixInt("-e:", 1024, "Number of edges (default:1024)\n");

  // tx model, alphas and betas
  const TModel Model = (TModel)Env.GetIfArgPrefixInt("-m:", 0, "Transmission model\n0:exponential, 1:power law, 2:rayleigh, 3:weibull (default:0)\n"); // tx model
  const TStr RAlphas = Env.GetIfArgPrefixStr("-ar:", TStr("0.01;1"), "Minimum and maximum alpha value (default:0.01;1)\n"); // alpha range
  const double k = Env.GetIfArgPrefixFlt("-k:", 1.0, "Shape parameter k for Weibull distribution (-m:3)\n"); // k for weibull
  const double Delta = Env.GetIfArgPrefixFlt("-d:", 1.0, "Delta for power-law (default:1)\n"); // delta for power law
  const double decayRatio = Env.GetIfArgPrefixFlt("-df:", 3.0, "Damping factor (default:3.0)\n");
  
  const int latentVariableSize  = Env.GetIfArgPrefixInt("-K:", 3, "Latent variable size");
  
  // num cascades, horizon per cascade & maximum time
  const int NCascades = Env.GetIfArgPrefixInt("-c:", 1000, "Number of cascades (default:1000)\n");
  const double Window = Env.GetIfArgPrefixFlt("-h:", 10.0, "Time horizon per cascade (default:10)\n");
  const double TotalTime = Env.GetIfArgPrefixFlt("-tt:", 100.0, "Total time (default:100)\n");

  // output filename
  const TStr FileName = Env.GetIfArgPrefixStr("-f:", TStr("example"), "Output name for network & cascades (default:example)\n");

  DecayCascadesModel decayCascades;

  decayCascades.SetTotalTime(TotalTime);
  decayCascades.SetWindow(Window);
  decayCascades.SetModel(Model);
  decayCascades.SetDelta(Delta);
  decayCascades.SetK(k);
  decayCascades.SetDecayRatio(decayRatio);
	
  TStrV RAlphasV; RAlphas.SplitOnAllCh(';', RAlphasV);
  TFlt MaxAlpha = RAlphasV[1].GetFlt(); 
  TFlt MinAlpha = RAlphasV[0].GetFlt();

  decayCascades.SetLatentVariableSize(latentVariableSize);
  decayCascades.SetMaxAlpha(MaxAlpha);
  decayCascades.SetMinAlpha(MinAlpha);

  // Generate network
  if (TNetwork<2) {
	  decayCascades.GenerateGroundTruth(TNetwork, NNodes, NEdges, NetworkParams); // Generate network
  } else {
	  TFIn GFIn(GroundTruthFileName);     // open network file
          InfoPathFileIO::LoadNetworkTxt(GFIn, decayCascades.Network, decayCascades.nodeInfo);
  }

  // Generate Cascades
  for (int i = 0; i < NCascades; i++) {
	  TCascade C(decayCascades.CascH.Len(), decayCascades.nodeInfo.Model);
	  decayCascades.GenCascade(C);
          decayCascades.CascH.AddDat(C.CId) = C;

	  printf("cascade:%d (%d nodes, first infection:%f, last infection:%f)\n", i, C.Len(), C.GetMinTm(), C.GetMaxTm());

	  // check the cascade last more than Window
	  IAssert( (C.GetMaxTm() - C.GetMinTm()) <= Window );
  }

  printf("Generate %d cascades!\n", decayCascades.GetCascs());

  if (TNetwork<2) decayCascades.SaveGroundTruth(FileName);
  InfoPathFileIO::SaveNetwork(TStr::Fmt("%s-network.txt", FileName.CStr()), decayCascades.Network, decayCascades.nodeInfo, decayCascades.edgeInfo);
  // Save Cascades
  InfoPathFileIO::SaveCascades(TStr::Fmt("%s-cascades.txt", FileName.CStr()), decayCascades.CascH, decayCascades.nodeInfo);
  decayCascades.SavePriorTopicProbability(TStr::Fmt("%s_PriorTopicProbability.txt",FileName.CStr()));

  Catch
  printf("\nrun time: %s (%s)\n", ExeTm.GetTmStr(), TSecTm::GetCurTm().GetTmStr().CStr());
  return 0;
}
