#ifndef FASTENMODEL_H
#define FASTENMODEL_H

#include <cascdynetinf.h>
#include <InfoPathFileIO.h>
#include <EM.h>
#include <FASTENFunction.h>
#include <TimeShapingFunction.h>

class FASTENModel {
   public:
      NodeInfo nodeInfo;
      EdgeInfo edgeInfo;
      THash<TInt, TCascade> CascH;
      TStrFltFltHNEDNet Network, InferredNetwork, MaxNetwork; 
      THash<TInt, THash<TIntPr,TFlt> > usedEdges;
      THash<TInt, THash<TInt,TInt> > outputEdgeMap;
     
      TFlt Window, TotalTime; 
      TFlt Delta, K;
      TFlt Gamma, Aging;

      FASTENFunctionConfigure fastenFunctionConfigure;
      FASTENFunction lossFunction;

      EMConfigure eMConfigure;
      EM<FASTENParameter> em;

      void LoadCascadesTxt(const TStr& InFNm);
      void LoadGroundTruthTxt(const TStr& InFNm);
      void SaveInferred(const TStr& OutFNm);
      void SavePriorTopicProbability(const TStr& OutFNm);
      void ReadPriorTopicProbability(const TStr& OutFNm);
      void ReadAlphas(const TStr& OutFNm);

      void GenCascade(TCascade& c);
      void GenerateGroundTruth(const int& TNetwork, const int& NNodes, const int& NEdges, const TStr& NetworkParams);
      void SaveGroundTruth(TStr);

      void SetLatentVariableSize(const TInt size) { fastenFunctionConfigure.latentVariableSize = eMConfigure.latentVariableSize = size;}
      void SetTotalTime(const float& tt) { TotalTime = tt; }
      void SetModel(const TModel& model) { nodeInfo.Model = model; }
      void SetWindow(const double& window) { Window = window; }
      void SetObservedWindow(const double& window) { lossFunction.observedWindow = window; }
      void SetDelta(const double& delta) { Delta = delta; }
      void SetK(const double& k) { K = k; }

      void SetLearningRate(const double& lr) { eMConfigure.pGDConfigure.learningRate = lr; }
      void SetBatchSize(const size_t batchSize) { eMConfigure.pGDConfigure.batchSize = batchSize;}
      void SetSampling(const TSampling sampling) { eMConfigure.pGDConfigure.sampling = sampling;} 
      void SetParamSampling(const TStr paramSampling) { eMConfigure.pGDConfigure.ParamSampling = paramSampling;}
      void SetMaxIterNm(const size_t maxIterNm) { eMConfigure.pGDConfigure.maxIterNm = maxIterNm;}
      void SetMaxEMIterNm(const size_t maxIterNm) { eMConfigure.maxIterNm = maxIterNm;}

      void SetAging(const double& aging) { Aging = aging; }
      void SetRegularizer(const TRegularizer& reg) { fastenFunctionConfigure.Regularizer = reg; }
      void SetMu(const double& mu) { fastenFunctionConfigure.Mu = mu; }
      void SetDecayRatio(const double& df) { fastenFunctionConfigure.decayRatio = df; }
      void SetTolerance(const double& tol) { fastenFunctionConfigure.Tol = tol; }
      void SetMaxAlpha(const double& ma) { fastenFunctionConfigure.MaxAlpha = edgeInfo.MaxAlpha = ma; }
      void SetMinAlpha(const double& ma) { fastenFunctionConfigure.MinAlpha = edgeInfo.MinAlpha = ma; }
      void SetInitAlpha(const double& ia) { fastenFunctionConfigure.InitAlpha = ia; }

      void Init();
      int GetCascs() { return CascH.Len(); }
      void Infer(const TFltV&, const TStr& OutFNm);
};

#endif
