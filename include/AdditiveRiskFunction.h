#ifndef ADDITIVERISKFUMCTION_H
#define ADDITIVERISKFUNCTION_H

#include <PGD.h>
#include <cascdynetinf.h>
#include <TimeShapingFunction.h>

typedef struct {
   TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
   TimeShapingFunction *shapingFunction;
   TRegularizer Regularizer;
   TFlt Mu, observedWindow;
}AdditiveRiskFunctionConfigure;

class AdditiveRiskFunction;

class AdditiveRiskParameter {
   friend class AdditiveRiskFunction;
   public:
      AdditiveRiskParameter();
      AdditiveRiskParameter& operator = (const AdditiveRiskParameter&);
      AdditiveRiskParameter& operator += (const AdditiveRiskParameter&);
      AdditiveRiskParameter& operator *= (const TFlt);
      AdditiveRiskParameter& projectedlyUpdateGradient(const AdditiveRiskParameter&);
      void reset();
      void set(AdditiveRiskFunctionConfigure configure);

      TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
      TRegularizer Regularizer;
      TFlt Mu;
      THash<TIntPr,TFlt> alphas;
};

class AdditiveRiskFunction : public PGDFunction<AdditiveRiskParameter> {
   public:
      void set(AdditiveRiskFunctionConfigure configure);
      AdditiveRiskParameter& gradient(Datum datum); 
      TFlt loss(Datum datum) const;
      void initPotentialEdges(Data);
      
      TimeShapingFunction *shapingFunction;
      TFlt observedWindow; 
      THash<TIntPr,TFlt> potentialEdges;
};

#endif 
