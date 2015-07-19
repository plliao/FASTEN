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
}NodeSoftMixCascadesFunctionConfigure;

class NodeSoftMixCascadesFunction;

class NodeSoftMixCascadesParameter {
   friend class NodeSoftMixCascadesFunction;
   public:
      NodeSoftMixCascadesParameter& operator = (const NodeSoftMixCascadesParameter&);
      NodeSoftMixCascadesParameter& operator += (const NodeSoftMixCascadesParameter&);
      NodeSoftMixCascadesParameter& operator *= (const TFlt);
      NodeSoftMixCascadesParameter& projectedlyUpdateGradient(const NodeSoftMixCascadesParameter&);
      void set(NodeSoftMixCascadesFunctionConfigure configure);
      void init(Data data, TInt NodeNm = 0);
      void initWeightParameter();
      void initAlphaParameter();
      void reset();

      TFlt GetTopicAlpha(TInt srcNId, TInt dstNId, TInt topic) const;
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt NId) const;

      TFlt Tol, InitAlpha, MaxAlpha, MinAlpha;
      TRegularizer Regularizer;
      TFlt Mu;
      TInt latentVariableSize;   
      THash<TInt, THash<TIntPr,TFlt> > kAlphas;
      THash<TInt, THash<TInt, TFlt> > nodeWeights;
};

class NodeSoftMixCascadesFunction : public PGDFunction<NodeSoftMixCascadesParameter> {
   public:
      TFlt loss(Datum datum) const;
      NodeSoftMixCascadesParameter& gradient(Datum datum);
      void calculateRMSProp(TFlt, NodeSoftMixCascadesParameter&, NodeSoftMixCascadesParameter&);
      void set(NodeSoftMixCascadesFunctionConfigure configure);
      void init(Data data, TInt NodeNm = 0);
      void initWeightParameter() { parameter.initWeightParameter();}
      void initAlphaParameter() { parameter.initAlphaParameter();}
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt NId) const { return parameter.GetAlpha(srcNId, dstNId, NId);}

      TimeShapingFunction *shapingFunction; 
      TFlt observedWindow;
};

#endif
