#ifndef MMRATEPARAMETER_H
#define MMRATEPARAMETER_H

#include <PGD.h>
#include <TimeShapingFunction.h>

typedef struct {
   TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
   TimeShapingFunction *shapingFunction;
   TRegularizer Regularizer;
   TFlt Mu;
   TInt latentVariableSize;   
}SoftMixCascadesFunctionConfigure;

class SoftMixCascadesFunction;

class SoftMixCascadesParameter {
   friend class SoftMixCascadesFunction;
   public:
      SoftMixCascadesParameter& operator = (const SoftMixCascadesParameter&);
      SoftMixCascadesParameter& operator += (const SoftMixCascadesParameter&);
      SoftMixCascadesParameter& operator *= (const TFlt);
      SoftMixCascadesParameter& projectedlyUpdateGradient(const SoftMixCascadesParameter&);
      void set(SoftMixCascadesFunctionConfigure configure);
      void init(Data data);
      void initParameter();
      void GenParameter();
      void GenCascadeWeight(TCascade&);
      void reset();

      TFlt GetTopicAlpha(TInt srcNId, TInt dstNId, TInt topic) const;
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt CId) const;

      TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
      TRegularizer Regularizer;
      TFlt Mu;
      TInt latentVariableSize;   
      THash<TInt, THash<TIntPr,TFlt> > kAlphas;
      THash<TInt, THash<TInt, TFlt> > cascadesWeights;
};

class SoftMixCascadesFunction : public PGDFunction<SoftMixCascadesParameter> {
   public:
      TFlt loss(Datum datum) const;
      SoftMixCascadesParameter& gradient(Datum datum);
      void set(SoftMixCascadesFunctionConfigure configure);
      void init(Data data);
      void initParameter() { parameter.initParameter();}
      void GenParameter() { parameter.GenParameter();}
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt CId) const { return parameter.GetAlpha(srcNId, dstNId, CId);}

      TimeShapingFunction *shapingFunction; 
      TFlt observedWindow;
};

#endif
