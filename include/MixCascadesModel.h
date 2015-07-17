#ifndef MMRATEMODEL_H
#define MMRATEMODEL_H

#include <cascdynetinf.h>
#include <InfoPathFileIO.h>
#include <EM.h>
#include <MixCascadesFunction.h>
#include <TimeShapingFunction.h>

class MixCascadesModel {
   public:
      NodeInfo nodeInfo;
      EdgeInfo edgeInfo;
      THash<TInt, TCascade> CascH;
      TStrFltFltHNEDNet Network, InferredNetwork; 
     
      TFlt Window, TotalTime, Delta; 
      TFlt Gamma, Aging;

      MixCascadesFunctionConfigure mixCascadesFunctionConfigure;
      MixCascadesFunction lossFunction;

      EMConfigure eMConfigure;
      EM<MixCascadesParameter> em;

      void LoadCascadesTxt(const TStr& InFNm);
      void LoadGroundTruthTxt(const TStr& InFNm);
      void SaveInferred(const TStr& OutFNm);

      void SetLatentVariableSize(const TInt size) { mixCascadesFunctionConfigure.latentVariableSize = eMConfigure.latentVariableSize = size;}
      void SetModel(const TModel& model) { nodeInfo.Model = model; }
      void SetWindow(const double& window) { Window = window; }
      void SetObservedWindow(const double& window) { mixCascadesFunctionConfigure.observedWindow = window; }
      void SetDelta(const double& delta) { Delta = delta; }

      void SetLearningRate(const double& lr) { eMConfigure.pGDConfigure.learningRate = lr; }
      void SetBatchSize(const size_t batchSize) { eMConfigure.pGDConfigure.batchSize = batchSize;}
      void SetSampling(const TSampling sampling) { eMConfigure.pGDConfigure.sampling = sampling;} 
      void SetParamSampling(const TStr paramSampling) { eMConfigure.pGDConfigure.ParamSampling = paramSampling;}
      void SetMaxIterNm(const size_t maxIterNm) { eMConfigure.pGDConfigure.maxIterNm = maxIterNm;}
      void SetEMMaxIterNm(const size_t maxIterNm) { eMConfigure.maxIterNm = maxIterNm;}

      void SetAging(const double& aging) { Aging = aging; }
      void SetRegularizer(const TRegularizer& reg) { mixCascadesFunctionConfigure.configure.Regularizer = reg; }
      void SetMu(const double& mu) { mixCascadesFunctionConfigure.configure.Mu = mu; }
      void SetTolerance(const double& tol) { mixCascadesFunctionConfigure.configure.Tol = tol; }
      void SetMaxAlpha(const double& ma) { mixCascadesFunctionConfigure.configure.MaxAlpha = edgeInfo.MaxAlpha = ma; }
      void SetMinAlpha(const double& ma) { mixCascadesFunctionConfigure.configure.MinAlpha = edgeInfo.MinAlpha = ma; }
      void SetInitAlpha(const double& ia) { mixCascadesFunctionConfigure.configure.InitAlpha = ia; }

      void Init();
      int GetCascs() { return CascH.Len(); }
      void Infer(const TFltV&, const TStr& OutFNm);
};

#endif
