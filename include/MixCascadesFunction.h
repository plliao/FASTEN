#ifndef MMRATEPARAMETER_H
#define MMRATEPARAMETER_H

#include <AdditiveRiskFunction.h>
#include <EM.h>

typedef struct {
   AdditiveRiskFunctionConfigure configure;
   TInt latentVariableSize;  
}MixCascadesFunctionConfigure;

class MixCascadesFunction;

class MixCascadesParameter {
   friend class MixCascadesFunction;
   public:
      MixCascadesParameter& operator = (const MixCascadesParameter&);
      MixCascadesParameter& operator += (const MixCascadesParameter&);
      MixCascadesParameter& operator *= (const TFlt);
      MixCascadesParameter& projectedlyUpdateGradient(const MixCascadesParameter&);
      void initKPiParameter();
      void init(TInt latentVariableSize);
      void set(MixCascadesFunctionConfigure configure);
      void reset();

      THash<TInt,TFlt> kPi, kPi_times;
      THash<TInt,AdditiveRiskFunction> kAlphas;
};

class MixCascadesFunction : public EMLikelihoodFunction<MixCascadesParameter> {
   public:
      TFlt JointLikelihood(Datum datum, TInt latentVariable) const;
      void maximize();
      MixCascadesParameter& gradient(Datum datum);
      void set(MixCascadesFunctionConfigure configure);
      void init(TInt latentVariableSize);
      void initKPiParameter();
      void initPotentialEdges(Data);

      THash<TIntPr,TFlt> potentialEdges;
};

#endif
