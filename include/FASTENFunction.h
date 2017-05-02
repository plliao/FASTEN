#ifndef FASTENPARAMETER_H
#define FASTENPARAMETER_H

#include <EM.h>
#include <TimeShapingFunction.h>

typedef struct {
   TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
   TimeShapingFunction *shapingFunction;
   TRegularizer Regularizer;
   TFlt Mu;
   TInt latentVariableSize;   
   TFlt decayRatio;
}FASTENFunctionConfigure;

class FASTENFunction;

class FASTENParameter {
   friend class FASTENFunction;
   public:
      FASTENParameter& operator = (const FASTENParameter&);
      FASTENParameter& operator += (const FASTENParameter&);
      FASTENParameter& operator *= (const TFlt);
      FASTENParameter& projectedlyUpdateGradient(const FASTENParameter&);
      void set(FASTENFunctionConfigure configure);
      void init(Data data, TInt NodeNm = 0);
      void initPriorTopicProbabilityParameter();
      void initAlphaParameter();
      void reset();

      TFlt GetTopicAlpha(TInt srcNId, TInt dstNId, TInt topic) const;
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt NId) const;

      TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
      TRegularizer Regularizer;
      TFlt Mu;
      TInt latentVariableSize;   
      THash<TInt, THash<TIntPr,TFlt> > kAlphas;
      THash<TInt, TFlt> priorTopicProbability;
      TFlt sampledTimes;
};

class FASTENFunction : public EMLikelihoodFunction<FASTENParameter> {
   public:
      TFlt JointLikelihood(Datum datum, TInt latentVariable) const;
      void maximize() ;
      FASTENParameter& gradient(Datum datum);
      void set(FASTENFunctionConfigure configure);
      void init(Data data, TInt NodeNm = 0);
      void initPriorTopicProbabilityParameter() { parameter.initPriorTopicProbabilityParameter();}
      void initAlphaParameter() { parameter.initAlphaParameter();}
      void initPotentialEdges(Data);
      TFlt GetTopicAlpha(TInt srcNId, TInt dstNId, TInt topic) const { return parameter.GetTopicAlpha(srcNId, dstNId, topic);}
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const { return parameter.GetAlpha(srcNId, dstNId, topic);}

      TimeShapingFunction *shapingFunction; 
      THash<TIntPr,TFlt> potentialEdges;
      TFlt observedWindow;
      TFlt decayRatio;
};

#endif
