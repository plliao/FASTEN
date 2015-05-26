#ifndef USERPROPERTYFUNCTION_H
#define USERPROPERTYFUNCTION_H

#include <UPEM.h>
#include <cascdynetinf.h>
#include <TimeShapingFunction.h>

typedef struct {
   TFlt propertyMinValue, propertyInitValue, propertyMaxValue;
   TFlt topicMinValue, topicInitValue, topicMaxValue;
   TFlt acquaintanceMinValue, acquaintanceInitValue, acquaintanceMaxValue;
   TFlt MaxAlpha, MinAlpha;
   TInt topicSize, propertySize;
   TimeShapingFunction *shapingFunction;
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
      void UpdateTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src, TFlt m, TFlt minValue, TFlt initValue, TFlt maxValue, TStr comment);
      void AddEqualTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src, TFlt m);
      void init(Data data, UserPropertyFunctionConfigure configure);
      void GenParameters(TStrFltFltHNEDNet& network, UserPropertyFunctionConfigure configure);

      TFlt propertyMinValue, propertyInitValue, propertyMaxValue;
      TFlt topicMinValue, topicInitValue, topicMaxValue;
      TFlt acquaintanceMinValue, acquaintanceInitValue, acquaintanceMaxValue;
      TFlt multiplier;
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
      UserPropertyParameter& gradient1(Datum datum);
      UserPropertyParameter& gradient2(Datum datum);
      UserPropertyParameter& gradient3(Datum datum);
      UserPropertyParameter& gradient(Datum datum);
      void initParameter(Data data, UserPropertyFunctionConfigure configure) {parameter.init(data,configure);}
      void GenParameters(TStrFltFltHNEDNet& network, UserPropertyFunctionConfigure configure) { set(configure); parameter.GenParameters(network, configure);}
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const;     
      TFlt GetAcquaitance(TInt srcNId, TInt dstNId) const;
      TFlt GetPropertyValue(TInt srcNId, TInt dstNId) const; 

      TInt propertySize;
      TFlt MaxAlpha, MinAlpha;
      TimeShapingFunction *shapingFunction; 
};

#endif 
