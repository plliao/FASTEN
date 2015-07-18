#ifndef MMRATEMODEL_H
#define MMRATEMODEL_H

#include <cascdynetinf.h>
#include <InfoPathFileIO.h>
#include <UPEM.h>
#include <UserPropertyFunction.h>
#include <TimeShapingFunction.h>
/*#ifdef max
#undef max
#endif 
#ifdef min
#undef min
#endif
#define ARMA_64BIT_WORD 
#include <armadillo>
using namespace arma;*/


class UserPropertyModel {
   public:
      NodeInfo nodeInfo;
      EdgeInfo edgeInfo;
      THash<TInt, TCascade> CascH;
      TStrFltFltHNEDNet Network, InferredNetwork; 
     
      TFlt Window, TotalTime; 
      TFlt Delta, K;
      TFlt Gamma, Aging;

      UserPropertyFunctionConfigure userPropertyFunctionConfigure;
      UserPropertyFunction lossFunction;

      UPEMConfigure eMConfigure;
      UPEM<UserPropertyParameter> em;

      void LoadCascadesTxt(const TStr& InFNm);
      void LoadGroundTruthTxt(const TStr& InFNm);
      void SaveInferred(const TStr& OutFNm);
      void SaveUserProperty(const TStr& OutFNm);
      void ReadUserProperty(const TStr& OutFNm);
      void SaveModel(const TStr& OutFNm);
      void ReadModel(const TStr& OutFNm);

      void GenParameters(TInt NEdges) { lossFunction.GenParameters(Network, userPropertyFunctionConfigure);}
      void GenCascade(TCascade& c);
      void GenerateGroundTruth(const int& TNetwork, const int& NNodes, const int& NEdges, const TStr& NetworkParams);
      void SaveGroundTruth(TStr);

      void SetLatentVariableSize(const TInt size) { userPropertyFunctionConfigure.parameter.topicSize = eMConfigure.latentVariableSize = size;}
      void SetPropertySize(const TInt size) { userPropertyFunctionConfigure.parameter.propertySize = size;}
      void SetTotalTime(const float& tt) { TotalTime = tt; }
      void SetModel(const TModel& model) { nodeInfo.Model = model; }
      void SetWindow(const double& window) { Window = window; }
      void SetDelta(const double& delta) { Delta = delta; }
      void SetK(const double& k) { K = k; }

      void SetLearningRate(const double& lr) { eMConfigure.pGDConfigure.learningRate = lr; }
      void SetRMSProbAlpha(const double& rmsAlpha) { eMConfigure.rmsAlpha = rmsAlpha; }
      void SetInitialMomentum(const double& imom) { eMConfigure.initialMomentum = imom; }
      void SetFinalMomentum(const double& fmom) { eMConfigure.finalMomentum = fmom; }
      void SetMomentumRatio(const double& momr) { eMConfigure.momentumRatio = momr; }
      void SetBatchSize(const size_t batchSize) { eMConfigure.pGDConfigure.batchSize = batchSize;}
      void SetSampling(const TSampling sampling) { eMConfigure.pGDConfigure.sampling = sampling;} 
      void SetParamSampling(const TStr paramSampling) { eMConfigure.pGDConfigure.ParamSampling = paramSampling;}
      void SetMaxIterNm(const size_t maxIterNm) { eMConfigure.pGDConfigure.maxIterNm = maxIterNm;}
      void SetEMMaxIterNm(const size_t maxIterNm) { eMConfigure.maxIterNm = maxIterNm;}
      void SetCOMaxIterNm(const size_t maxIterNm) { eMConfigure.maxCoorIterNm = maxIterNm;}

      void SetAging(const double& aging) { Aging = aging; }
      void SetRegularizer(const TRegularizer& reg) { userPropertyFunctionConfigure.parameter.Regularizer = reg; }
      void SetMu(const double& mu) { userPropertyFunctionConfigure.parameter.Mu = mu; }
      
      void SetMaxAlpha(const double& ma) {userPropertyFunctionConfigure.parameter.MaxAlpha = edgeInfo.MaxAlpha = ma;} 
      void SetMinAlpha(const double& ma) {userPropertyFunctionConfigure.parameter.MinAlpha = edgeInfo.MinAlpha = ma;} 

      void Init();
      int GetCascs() { return CascH.Len(); }
      void Infer(const TFltV&, const TStr& OutFNm);
      //void ExtractFeature();
      //fmat sigmoid(fmat&);
};

#endif
