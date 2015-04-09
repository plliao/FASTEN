#ifndef RNCASCADE_H
#define RNCASCADE_H

#include <cascdynetinf.h>

class RNCascade : public TNIBs {
   public:
     TVec<TStrFltFltHNEDNet> MultipleInferredNetworks;
     TVec<TIntFltH> KAveDiffAlphas;
     TVec<TInt> RNNIds;

     TFlt EntropyThreshold; 
     TInt RNNumber;

   public:
     RNCascade() {}

     void LoadMultipleGroundTruthTxt(TSIn &SIn);
     void LoadMultipleCascadesTxt(TSIn &SIn);

     void SetEntropyThreshold(const double& entropyThreshold) { EntropyThreshold = entropyThreshold;}
     void SetRNNumber(const int& rnn) { RNNumber = rnn;}

     void Reset();
     void Init(const TFltV& Steps);

     void SG(const int& NId, const int& Iters, const TFltV& Steps, const TSampling& Sampling, const TStr& ParamSampling=TStr(""), const bool& PlotPerformance=false);
     void BSG(const int& NId, const int& Iters, const TFltV& Steps, const int& BatchLen, const TSampling& Sampling, const TStr& ParamSampling=TStr(""), const bool& PlotPerformance=false);
     void FG(const int& NId, const int& Iters, const TFltV& Steps);
  
     void UpdateDiff(const TOptMethod& OptMethod, const int& NId, TCascade& Cascade, THash<TInt,TIntPrV>& KAlphasToUpdate, const double& CurrentTime=TFlt::Mx);
     

     void SaveMultipleInferred(const TStr& OutFNm, const TIntV& NIdV=TIntV());

     void GenerateInferredNetwork(const int& NId);
     void GenerateRNNIds();

     TFlt crossEntropy(const TInt &NId1, const TInt &NId2) const;
     bool isRNCategory(const int& NId, TCascade& Cascade);
};

#endif
