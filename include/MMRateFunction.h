#ifndef MMRATEPARAMETER_H
#define MMRATEPARAMETER_H

#include <AdditiveRiskFunction.h>
#include <EM.h>

typedef struct {
   AdditiveRiskFunctionConfigure configure;
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
      void init(TInt latentVariableSize, THash<TInt,AdditiveRiskFunction>* kAlphasP);
      void set(MMRateFunctionConfigure configure);
      void reset();
      THash<TInt,AdditiveRiskFunction>& getKAlphas() { return kAlphas;}
      THash<TInt,TFlt>& getKPi() { return kPi;}
   private:
      THash<TInt,TFlt> kPi, kPi_times;
      THash<TInt,AdditiveRiskFunction> kAlphas;
      THash<TInt,AdditiveRiskFunction>* kAlphasP;
};

class MMRateFunction : public EMLikelihoodFunction<MMRateParameter> {
   public:
      TFlt JointLikelihood(Datum datum, TInt latentVariable) const;
      void maximize();
      MMRateParameter& gradient(Datum datum);
      void set(MMRateFunctionConfigure configure);
      THash<TInt,AdditiveRiskFunction>& getKAlphas() { return kAlphas;}
      THash<TInt,TFlt>& getKPi() { return getParameter().getKPi();}
   private:
      THash<TInt,AdditiveRiskFunction> kAlphas;
};

#endif
