#ifndef MMRATEMODEL_H
#define MMRATEMODEL_H

#include <cascdynetinf.h>
#include <InfoPathFileIO.h>
#include <PGD.h>
#include <EM.h>
#include <SoftMixCascadesFunction.h>
#include <TimeShapingFunction.h>

class SoftMixCascadesModel {
   public:
      NodeInfo nodeInfo;
      EdgeInfo edgeInfo;
      THash<TInt, TCascade> CascH;
      TStrFltFltHNEDNet Network, InferredNetwork; 
     
      TFlt Window, TotalTime; 
      TFlt Delta, K;
      TFlt Gamma, Aging;

      SoftMixCascadesFunctionConfigure softMixCascadesFunctionConfigure;
      SoftMixCascadesFunction lossFunction;

      EMConfigure eMConfigure;
      PGD<SoftMixCascadesParameter> pgd;

      void LoadCascadesTxt(const TStr& InFNm);
      void LoadGroundTruthTxt(const TStr& InFNm);
      void SaveInferred(const TStr& OutFNm);
      void SaveWeights(const TStr& OutFNm);

      void GenCascade(TCascade& c);
      void GenerateGroundTruth(const int& TNetwork, const int& NNodes, const int& NEdges, const TStr& NetworkParams);
      void SaveGroundTruth(TStr);

      void SetLatentVariableSize(const TInt size) { softMixCascadesFunctionConfigure.latentVariableSize = eMConfigure.latentVariableSize = size;}
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

      void SetAging(const double& aging) { Aging = aging; }
      void SetRegularizer(const TRegularizer& reg) { softMixCascadesFunctionConfigure.Regularizer = reg; }
      void SetMu(const double& mu) { softMixCascadesFunctionConfigure.Mu = mu; }
      void SetTolerance(const double& tol) { softMixCascadesFunctionConfigure.Tol = tol; }
      void SetMaxAlpha(const double& ma) { softMixCascadesFunctionConfigure.MaxAlpha = edgeInfo.MaxAlpha = ma; }
      void SetMinAlpha(const double& ma) { softMixCascadesFunctionConfigure.MinAlpha = edgeInfo.MinAlpha = ma; }
      void SetInitAlpha(const double& ia) { softMixCascadesFunctionConfigure.InitAlpha = ia; }

      void Init();
      int GetCascs() { return CascH.Len(); }
      void Infer(const TFltV&, const TStr& OutFNm);
};

#endif
