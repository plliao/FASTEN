#ifndef USERPROPERTYFUNCTION_H
#define USERPROPERTYFUNCTION_H

#include <UPEM.h>
#include <cascdynetinf.h>
#include <TimeShapingFunction.h>

typedef struct {
   TFlt propertyMinValue, propertyInitValue, propertyMaxValue;
   TFlt topicMinValue, topicInitValue, topicMaxValue, topicStdValue;
   TFlt acquaintanceMinValue, acquaintanceInitValue, acquaintanceMaxValue;
   TFlt MaxAlpha, MinAlpha;
   TInt topicSize, propertySize;
   TimeShapingFunction *shapingFunction;
   TRegularizer Regularizer;
   TFlt Mu;
}UserPropertyFunctionConfigure;

class UserPropertyFunction;

class UserPropertyParameter {
   friend class UserPropertyFunction;
   public:
      UserPropertyParameter();
      UserPropertyParameter& operator = (const UserPropertyParameter&);
      UserPropertyParameter& operator += (const UserPropertyParameter&);
      UserPropertyParameter& operator *= (const TFlt);
      UserPropertyParameter& projectedlyUpdateGradient(const UserPropertyParameter&);
      void reset();
      void set(UserPropertyFunctionConfigure configure);
      void UpdateTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src, TFlt minValue, TFlt initValue, TFlt maxValue, TStr comment);
      void AddEqualTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src);
      void MultiplyTHash(THash<TIntPr,TFlt>& dst, const TFlt multiplier);
      void init(Data data, UserPropertyFunctionConfigure configure);
      void GenParameters(TStrFltFltHNEDNet& network, UserPropertyFunctionConfigure configure, TInt);
      
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const;     
      TFlt GetAcquaitance(TInt srcNId, TInt dstNId) const;
      TFlt GetPropertyValue(TInt srcNId, TInt dstNId) const;
      TFlt GetTopicValue(TInt srcNId, TInt dstNId, TInt topic) const; 

      TFlt propertyMinValue, propertyInitValue, propertyMaxValue;
      TFlt topicMinValue, topicInitValue, topicMaxValue, topicStdValue;
      //TFlt acquaintanceMinValue, acquaintanceInitValue, acquaintanceMaxValue;
      TFlt MaxAlpha, MinAlpha;
      TInt propertySize;
      TRegularizer Regularizer;
      TFlt Mu;
      THash<TIntPr,TFlt> acquaintance;
      THash<TIntPr,TFlt> receiverProperty, spreaderProperty;
      THash<TIntPr,TFlt> topicReceive, topicSpread; 
      THash<TInt,TFlt> kPi, kPi_times;
};

class UserPropertyFunction : public UPEMLikelihoodFunction<UserPropertyParameter> {
   public:
      void set(UserPropertyFunctionConfigure configure);
      TFlt JointLikelihood(Datum datum, TInt latentVariable) const;
      void maximize();
      void updateAcquaintance();
      UserPropertyParameter& gradient(Datum datum);
      void calculateRProp(TFlt, UserPropertyParameter&, UserPropertyParameter&);
      void calculateRMSProp(TFlt, UserPropertyParameter&, UserPropertyParameter&);
      void calculateAverageRMSProp(TFlt, TFltV&, UserPropertyParameter&);
      void initParameter(Data data, UserPropertyFunctionConfigure configure);
      void GenParameters(TStrFltFltHNEDNet& network, UserPropertyFunctionConfigure configure, TInt edgeNums) { set(configure); parameter.GenParameters(network, configure, edgeNums);}
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const { return parameter.GetAlpha(srcNId, dstNId, topic);}     
      TFlt GetAcquaitance(TInt srcNId, TInt dstNId) const { return parameter.GetAcquaitance(srcNId, dstNId);}
      TFlt GetPropertyValue(TInt srcNId, TInt dstNId) const { return parameter.GetPropertyValue(srcNId, dstNId);}
      TFlt GetTopicValue(TInt srcNId, TInt dstNId, TInt topic) const { return parameter.GetTopicValue(srcNId, dstNId, topic);}
      THash<TInt,TFlt> getPriorTHash() const { return parameter.kPi;} 

      TInt propertySize;
      TFlt MaxAlpha, MinAlpha;
      TimeShapingFunction *shapingFunction; 
};

#endif 
