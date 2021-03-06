#ifndef MMRATEMODEL_H
#define MMRATEMODEL_H

#include <cascdynetinf.h>
#include <InfoPathFileIO.h>
#include <EM.h>
#include <MMRateFunction.h>
#include <TimeShapingFunction.h>

class MMRateModel {
   public:
      NodeInfo nodeInfo;
      EdgeInfo edgeInfo;
      THash<TInt, TCascade> CascH;
      TStrFltFltHNEDNet Network, InferredNetwork, MaxNetwork; 
     
      TFlt Window, TotalTime, Delta; 
      TFlt Gamma, Aging;

      MMRateFunctionConfigure mMRateFunctionConfigure;
      MMRateFunction lossFunction;

      EMConfigure eMConfigure;
      EM<MMRateParameter> em;

      void LoadCascadesTxt(const TStr& InFNm);
      void LoadGroundTruthTxt(const TStr& InFNm);
      void SaveInferred(const TStr& OutFNm);
      void SaveDiffusionPatterns(const TStr& OutFNm);

      void SetLatentVariableSize(const TInt size) { mMRateFunctionConfigure.latentVariableSize = eMConfigure.latentVariableSize = size;}
      void SetModel(const TModel& model) { nodeInfo.Model = model; }
      void SetWindow(const double& window) { Window = window; }
      void SetObservedWindow(const double& window) { lossFunction.observedWindow = window; }
      void SetDelta(const double& delta) { Delta = delta; }

      void SetLearningRate(const double& lr) { eMConfigure.pGDConfigure.learningRate = lr; }
      void SetBatchSize(const size_t batchSize) { eMConfigure.pGDConfigure.batchSize = batchSize;}
      void SetSampling(const TSampling sampling) { eMConfigure.pGDConfigure.sampling = sampling;} 
      void SetParamSampling(const TStr paramSampling) { eMConfigure.pGDConfigure.ParamSampling = paramSampling;}
      void SetMaxIterNm(const size_t maxIterNm) { eMConfigure.pGDConfigure.maxIterNm = maxIterNm;}
      void SetEMMaxIterNm(const size_t maxIterNm) { eMConfigure.maxIterNm = maxIterNm;}

      void SetAging(const double& aging) { Aging = aging; }
      void SetRegularizer(const TRegularizer& reg) { mMRateFunctionConfigure.Regularizer = reg; }
      void SetMu(const double& mu) { mMRateFunctionConfigure.Mu = mu; }
      void SetTolerance(const double& tol) { mMRateFunctionConfigure.Tol = tol; }
      void SetMaxAlpha(const double& ma) { mMRateFunctionConfigure.MaxAlpha = edgeInfo.MaxAlpha = ma; }
      void SetMinAlpha(const double& ma) { mMRateFunctionConfigure.MinAlpha = edgeInfo.MinAlpha = ma; }
      void SetInitAlpha(const double& ia) { mMRateFunctionConfigure.InitAlpha = ia; }
      void SetMaxDiffusionPattern(const double& md) { mMRateFunctionConfigure.MaxDiffusionPattern = md; }
      void SetMinDiffusionPattern(const double& md) { mMRateFunctionConfigure.MinDiffusionPattern = md; }
      void SetInitDiffusionPattern(const double& id) { mMRateFunctionConfigure.InitDiffusionPattern = id; }

      void Init();
      int GetCascs() { return CascH.Len(); }
      void Infer(const TFltV&, const TStr& OutFNm);
};

#endif
