#ifndef MMRATEPARAMETER_H
#define MMRATEPARAMETER_H

#include <EM.h>
#include <TimeShapingFunction.h>

typedef struct {
   TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
   TFlt InitDiffusionPattern, MaxDiffusionPattern, MinDiffusionPattern;
   TimeShapingFunction *shapingFunction;
   TRegularizer Regularizer;
   TFlt Mu;
   TInt latentVariableSize;   
}MMRateFunctionConfigure;

class MMRateFunction;

class MMRateParameter {
   friend class MMRateFunction;
   public:
      MMRateParameter& operator = (const MMRateParameter&);
      MMRateParameter& operator += (const MMRateParameter&);
      MMRateParameter& operator *= (const TFlt);
      MMRateParameter& projectedlyUpdateGradient(const MMRateParameter&);
      void set(MMRateFunctionConfigure configure);
      void reset();

      TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
      TFlt InitDiffusionPattern, MaxDiffusionPattern, MinDiffusionPattern;
      TRegularizer Regularizer;
      TFlt Mu;
      TInt latentVariableSize;   
      THash<TInt, THash<TIntPr,TFlt> > kAlphas;
      THash<TInt,TFlt> diffusionPatterns; 
      THash<TInt,TFlt> kPi, kPi_times;
};

class MMRateFunction : public EMLikelihoodFunction<MMRateParameter> {
   public:
      TFlt JointLikelihood(Datum datum, TInt latentVariable) const;
      void maximize();
      MMRateParameter& gradient(Datum datum);
      void set(MMRateFunctionConfigure configure);
      void initPotentialEdges(Data);

      TimeShapingFunction *shapingFunction; 
      TFlt observedWindow;
      THash<TIntPr,TFlt> potentialEdges;
};

#endif
