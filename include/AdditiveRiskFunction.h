#ifndef ADDITIVERISKFUMCTION_H
#define ADDITIVERISKFUNCTION_H

#include <PGD.h>
#include <cascdynetinf.h>
#include <TimeShapingFunction.h>

typedef struct {
   TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
   TimeShapingFunction *shapingFunction;
   TRegularizer Regularizer;
   TFlt Mu;
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
      const THash<TIntPr,TFlt>& getAlphas() const;
      const THash<TInt,TFlt>& getInitialAlphas() const;
      const TFlt getMultiplier() const;
   private:
      TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
      TFlt multiplier;
      TRegularizer Regularizer;
      TFlt Mu;
      THash<TIntPr,TFlt> alphas;
      THash<TInt,TFlt> initialAlphas; 
};

class AdditiveRiskFunction : public PGDFunction<AdditiveRiskParameter> {
   public:
      void set(AdditiveRiskFunctionConfigure configure);
      AdditiveRiskParameter& gradient(Datum datum); 
      TFlt loss(Datum datum) const;
      
      TimeShapingFunction *shapingFunction; 
};

#endif 
