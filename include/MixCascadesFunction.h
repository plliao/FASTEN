#ifndef MMRATEPARAMETER_H
#define MMRATEPARAMETER_H

#include <AdditiveRiskFunction.h>
#include <EM.h>

typedef struct {
   AdditiveRiskFunctionConfigure configure;
   TInt latentVariableSize;  
   TFlt observedWindow; 
}MixCascadesFunctionConfigure;

class MixCascadesFunction;

class MixCascadesParameter {
   friend class MixCascadesFunction;
   public:
      MixCascadesParameter& operator = (const MixCascadesParameter&);
      MixCascadesParameter& operator += (const MixCascadesParameter&);
      MixCascadesParameter& operator *= (const TFlt);
      MixCascadesParameter& projectedlyUpdateGradient(const MixCascadesParameter&);
      void init(TInt latentVariableSize, THash<TInt,AdditiveRiskFunction>* kAlphasP);
      void set(MixCascadesFunctionConfigure configure);
      void reset();
      THash<TInt,AdditiveRiskFunction>& getKAlphas() { return kAlphas;}
      THash<TInt,TFlt>& getKPi() { return kPi;}
   private:
      THash<TInt,TFlt> kPi, kPi_times;
      THash<TInt,AdditiveRiskFunction> kAlphas;
      THash<TInt,AdditiveRiskFunction>* kAlphasP;
};

class MixCascadesFunction : public EMLikelihoodFunction<MixCascadesParameter> {
   public:
      TFlt JointLikelihood(Datum datum, TInt latentVariable) const;
      void maximize();
      MixCascadesParameter& gradient(Datum datum);
      void set(MixCascadesFunctionConfigure configure);
      THash<TInt,AdditiveRiskFunction>& getKAlphas() { return kAlphas;}
      THash<TInt,TFlt>& getKPi() { return getParameter().getKPi();}
   private:
      THash<TInt,AdditiveRiskFunction> kAlphas;
};

#endif
