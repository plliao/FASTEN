#ifndef MMRATEMODEL_H
#define MMRATEMODEL_H

#include <cascdynetinf.h>
#include <InfoPathFileIO.h>
#include <UPEM.h>
#include <UserPropertyFunction.h>
#include <TimeShapingFunction.h>
#ifdef max
#undef max
#endif 
#ifdef min
#undef min
#endif 
#include <armadillo>
using namespace arma;

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

      void GenParameters() { lossFunction.GenParameters(Network,userPropertyFunctionConfigure);}
      void GenCascade(TCascade& c);
      void GenerateGroundTruth(const int& TNetwork, const int& NNodes, const int& NEdges, const TStr& NetworkParams);
      void SaveGroundTruth(TStr);

      void SetLatentVariableSize(const TInt size) { userPropertyFunctionConfigure.topicSize = eMConfigure.latentVariableSize = size;}
      void SetPropertySize(const TInt size) { userPropertyFunctionConfigure.propertySize = size;}
      void SetTotalTime(const float& tt) { TotalTime = tt; }
      void SetModel(const TModel& model) { nodeInfo.Model = model; }
      void SetWindow(const double& window) { Window = window; }
      void SetDelta(const double& delta) { Delta = delta; }
      void SetK(const double& k) { K = k; }

      void SetLearningRate(const double& lr) { eMConfigure.pGDConfigure.learningRate = lr; }
      void SetBatchSize(const size_t batchSize) { eMConfigure.pGDConfigure.batchSize = batchSize;}
      void SetSampling(const TSampling sampling) { eMConfigure.pGDConfigure.sampling = sampling;} 
      void SetParamSampling(const TStr paramSampling) { eMConfigure.pGDConfigure.ParamSampling = paramSampling;}
      void SetMaxIterNm(const size_t maxIterNm) { eMConfigure.pGDConfigure.maxIterNm = maxIterNm;}
      void SetEMMaxIterNm(const size_t maxIterNm) { eMConfigure.maxIterNm = maxIterNm;}
      void SetCOMaxIterNm(const size_t maxIterNm) { eMConfigure.maxCoorIterNm = maxIterNm;}

      void SetAging(const double& aging) { Aging = aging; }
      void SetRegularizer(const TRegularizer& reg) { userPropertyFunctionConfigure.Regularizer = reg; }
      void SetMu(const double& mu) { userPropertyFunctionConfigure.Mu = mu; }
      
      void SetAcquaintanceMinValue(const double& mv) { userPropertyFunctionConfigure.acquaintanceMinValue = mv; }
      void SetAcquaintanceMaxValue(const double& mv) { userPropertyFunctionConfigure.acquaintanceMaxValue = mv; }
      void SetAcquaintanceInitValue(const double& iv) { userPropertyFunctionConfigure.acquaintanceInitValue = iv; }
      
      void SetPropertyMinValue(const double& mv) { userPropertyFunctionConfigure.propertyMinValue = mv; }
      void SetPropertyMaxValue(const double& mv) { userPropertyFunctionConfigure.propertyMaxValue = mv; }
      void SetPropertyInitValue(const double& iv) { userPropertyFunctionConfigure.propertyInitValue = iv; }
      
      void SetTopicMinValue(const double& mv) { userPropertyFunctionConfigure.topicMinValue = mv; }
      void SetTopicMaxValue(const double& mv) { userPropertyFunctionConfigure.topicMaxValue = mv; }
      void SetTopicInitValue(const double& iv) { userPropertyFunctionConfigure.topicInitValue = iv; }
      void SetTopicStdValue(const double& sv) { userPropertyFunctionConfigure.topicStdValue = sv; }

      void SetMaxAlpha(const double& ma) {userPropertyFunctionConfigure.MaxAlpha = edgeInfo.MaxAlpha = ma;} 
      void SetMinAlpha(const double& ma) {userPropertyFunctionConfigure.MinAlpha = edgeInfo.MinAlpha = ma;} 

      void Init();
      int GetCascs() { return CascH.Len(); }
      void Infer(const TFltV&, const TStr& OutFNm);
      void ExtractFeature();
      fmat sigmoid(fmat&);
};

#endif
