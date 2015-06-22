#ifndef INFO_H
#define INFO_H

#include <cascdynetinf.h>
#include <InfoPathFileIO.h>
#include <PGD.h>
#include <AdditiveRiskFunction.h>
#include <TimeShapingFunction.h>

class Info {
   public:
      NodeInfo nodeInfo;
      EdgeInfo edgeInfo;
      THash<TInt, TCascade> CascH;
      TStrFltFltHNEDNet Network, InferredNetwork;
     
      TFlt Window, TotalTime, Delta; 
      TFlt Gamma, Mu, Aging;
      TRegularizer Regularizer;

      AdditiveRiskFunctionConfigure additiveRiskFunctionConfigure;
      AdditiveRiskFunction lossFunction;

      PGDConfigure pGDConfigure;
      PGD<AdditiveRiskParameter> pgd;

      void LoadCascadesTxt(const TStr& InFNm);
      void LoadGroundTruthTxt(const TStr& InFNm);
      void SaveInferred(const TStr& OutFNm);

      void SetModel(const TModel& model) { nodeInfo.Model = model; }
      void SetWindow(const double& window) { Window = window; }
      void SetDelta(const double& delta) { Delta = delta; }

      void SetLearningRate(const double& lr) { pGDConfigure.learningRate = lr; }
      void SetBatchSize(const size_t batchSize) { pGDConfigure.batchSize = batchSize;}
      void SetSampling(const TSampling sampling) {pGDConfigure.sampling = sampling;} 
      void SetParamSampling(const TStr paramSampling) {pGDConfigure.ParamSampling = paramSampling;}
      void SetMaxIterNm(const size_t maxIterNm) { pGDConfigure.maxIterNm = maxIterNm;}

      void SetAging(const double& aging) { Aging = aging; }
      void SetRegularizer(const TRegularizer& reg) { Regularizer = reg; }
      void SetMu(const double& mu) { Mu = mu; }
      void SetTolerance(const double& tol) { additiveRiskFunctionConfigure.Tol = tol; }
      void SetMaxAlpha(const double& ma) { additiveRiskFunctionConfigure.MaxAlpha = edgeInfo.MaxAlpha = ma; }
      void SetMinAlpha(const double& ma) { additiveRiskFunctionConfigure.MinAlpha = edgeInfo.MinAlpha = ma; }
      void SetInitAlpha(const double& ia) { additiveRiskFunctionConfigure.InitAlpha = ia; }

      void Init();
      int GetCascs() { return CascH.Len(); }
      void Infer(const TFltV&);
};

#endif
