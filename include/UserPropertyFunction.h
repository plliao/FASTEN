#ifndef USERPROPERTYFUNCTION_H
#define USERPROPERTYFUNCTION_H

#include <UPEM.h>
#include <cascdynetinf.h>
#include <TimeShapingFunction.h>

typedef struct {
   TFlt MaxAlpha, MinAlpha;
   TInt topicSize, propertySize;
   TRegularizer Regularizer;
   TFlt Mu;
}UserPropertyParameterConfigure;

typedef struct {
   TimeShapingFunction *shapingFunction;
   UserPropertyParameterConfigure parameter;
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
      //void UpdateTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src, TFlt minValue, TFlt initValue, TFlt maxValue, TStr comment);
      //void AddEqualTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src);
      //void MultiplyTHash(THash<TIntPr,TFlt>& dst, const TFlt multiplier);
      void init(Data data);
      void GenParameters(TStrFltFltHNEDNet& Network);
      
      TFlt GetValue(TInt srcNId, TInt dstNId, TInt topic) const;    
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const;    

      UserPropertyParameterConfigure configure; 
      THash<TInt, THash<TInt,TFlt> > kWeights;
      THash<TIntPr,TFlt> receiverProperty, spreaderProperty;
      THash<TInt,TFlt> kPi, kPi_times;
};

class UserPropertyFunction : public UPEMLikelihoodFunction<UserPropertyParameter> {
   public:
      void set(UserPropertyFunctionConfigure configure);
      TFlt JointLikelihood(Datum datum, TInt latentVariable) const;
      void maximize();
      UserPropertyParameter& gradient(Datum datum);
      void calculateRProp(TFlt, UserPropertyParameter&, UserPropertyParameter&);
      void calculateRMSProp(TFlt, UserPropertyParameter&, UserPropertyParameter&);
      void calculateAverageRMSProp(TFlt, TFltV&, UserPropertyParameter&);
      void initParameter(Data data, UserPropertyFunctionConfigure configure);
      void GenParameters(TStrFltFltHNEDNet& Network, UserPropertyFunctionConfigure configure) { set(configure); parameter.GenParameters(Network);}
      TFlt GetValue(TInt srcNId, TInt dstNId, TInt topic) const { return parameter.GetValue(srcNId, dstNId, topic);}     
      TFlt GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const { return parameter.GetAlpha(srcNId, dstNId, topic);}     
      THash<TInt,TFlt> getPriorTHash() const { return parameter.kPi;} 

      TInt propertySize;
      TFlt MaxAlpha, MinAlpha;
      TimeShapingFunction *shapingFunction; 
};

#endif 
