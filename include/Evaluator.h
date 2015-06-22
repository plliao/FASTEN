#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <cascdynetinf.h>

typedef TFltPr PRCPoint;
typedef TVec<PRCPoint> PRCPoints;
typedef THash<TFlt,PRCPoints> DyPRCPoints;

class Evaluator {
  public:
    
    void LoadGroundTruth(TSIn &SIn);
    void LoadInferredNetwork(TSIn &SIn, TStr modelName);
    void EvaluatePRC(TFlt minAlpha, TFlt maxAlpha, const TFlt &step, TFlt PRCPointNm=100.0, bool verbol=true);
    void EvaluateAUC(const TFlt &step);
    void EvaluateMSE(const TFlt &step);
    void PlotPRC(const TStr &Str) const;
    void PlotMSE(const TStr &Str) const;

    TVec<TFlt> GetSteps(size_t i) const;
    TFlt GetGroundTruthTimeStep(TFlt step) const;
    TFlt GetInferredTimeStep(TFlt step, size_t i) const;

  public:
    TStrFltFltHNEDNet GroundTruth;  
    TVec<TStrFltFltHNEDNet> InferredNetworks;
    TVec<TStr> ModelNames;
    TVec<DyPRCPoints> PRC; 
    TVec<TFltFltH> MSE, MAE, PRC_AUC;

};

#endif
