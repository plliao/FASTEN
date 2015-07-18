#include <UserPropertyFunction.h>
#include <ctime>

TFlt sigmoid(TFlt t) {
   if (t > 1000.0) t = 1000.0;
   if (t < -1000.0) t = -1000.0;
   return 1.0/(1.0 + TMath::Power(TMath::E,-1.0*t));
}

TFlt ReLU(TFlt t) {
   if (t < -10.0) t = -10.0;
   return TMath::Log(1.0 + TMath::Power(TMath::E,t));
}

void UserPropertyFunction::set(UserPropertyFunctionConfigure configure) {
   shapingFunction = configure.shapingFunction;
   topicSize = configure.parameter.topicSize;
   propertySize = configure.parameter.propertySize;
   MaxAlpha = configure.parameter.MaxAlpha;
   MinAlpha = configure.parameter.MinAlpha;
   parameter.set(configure);
   parameterGrad.set(configure);
}

TFlt UserPropertyFunction::JointLikelihood(Datum datum, TInt latentVariable) const {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
   TFlt totalLoss = 0.0;

   int nodeSize = NodeNmH.Len();
   float *lossTable = new float[nodeSize];
   #pragma omp parallel for
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      TInt dstNId = key, srcNId;
      TFlt sumInLog = 0.0, val = 0.0;
      TFlt dstTime, srcTime;

      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) dstTime = Cascade.GetTm(dstNId);
      else dstTime = CurrentTime;

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
                        
         TFlt alpha = GetAlpha(srcNId, dstNId, latentVariable);
 
         sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
         val += alpha * shapingFunction->Integral(srcTime,dstTime);
         /*printf("datum:%d, topic:%d, sumInLog:%f, val:%f, alpha:%f, shapingVal:%f, shapingInt:%f\n", \
                datum.index(), latentVariable(), sumInLog(),val(),alpha(),shapingFunction->Value(srcTime,dstTime)(),shapingFunction->Integral(srcTime,dstTime)()); */
      }
      lossTable[i] = val;
      
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime && sumInLog != 0.0) lossTable[i] -= TMath::Log(sumInLog);
   }

   for (int i=0;i<nodeSize;i++) totalLoss += lossTable[i];
   delete[] lossTable;

   TFlt logP = -1.0 * totalLoss;
   TFlt logPi = TMath::Log(parameter.kPi.GetDat(latentVariable));
   //printf("datum:%d, topic:%d, logP: %f, logPi=%f\n",datum.index(), latentVariable(), logP(), logPi());
   return logP + logPi;
}

UserPropertyParameter& UserPropertyFunction::gradient(Datum datum) {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;

   parameterGrad.reset();
   
   int nodeSize = NodeNmH.Len();

   #pragma omp parallel for
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      TInt dstNId = key, srcNId;
      TFlt dstTime, srcTime;

      TIntPr receiverIndex, spreaderIndex;
      
      bool inCascade = false;
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) {
         dstTime = Cascade.GetTm(dstNId);
         inCascade = true;
      }
      else dstTime = CurrentTime;

      THash<TInt,TFlt> dstAlphaVector;
      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
         
         for (TInt latentVariable=0; latentVariable < topicSize; latentVariable++) {
            if (inCascade) {
               TFlt hazard = GetAlpha(srcNId, dstNId, latentVariable) * shapingFunction->Value(srcTime,dstTime);
               if (!dstAlphaVector.IsKey(latentVariable)) dstAlphaVector.AddDat(latentVariable,hazard);
               else dstAlphaVector.GetDat(latentVariable) += hazard;
            }
         }
      }

      THash<TInt, THash<TInt,TFlt> > kWeightGradient;
      for (TInt latentVariable=0; latentVariable < topicSize; latentVariable++) {
         kWeightGradient.AddDat(latentVariable, THash<TInt,TFlt>());
      }
 
      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
            
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         for (TInt latentVariable=0; latentVariable < topicSize; latentVariable++) {
            TFlt value =  GetValue(srcNId, dstNId, latentVariable);
            THash<TInt,TFlt>& weight = kWeightGradient.GetDat(latentVariable);

            for (TInt propertyIndex=0; propertyIndex < propertySize; propertyIndex++) {
               spreaderIndex.Val2 = receiverIndex.Val2 = propertyIndex;
               TFlt spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);
               TFlt receiverValue = parameter.receiverProperty.GetDat(receiverIndex);
         
               TFlt grad;
               if (inCascade) {
                  TFlt totalAlpha = dstAlphaVector.GetDat(latentVariable);
                  grad = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/totalAlpha;
               }
               else {
                  grad =  shapingFunction->Integral(srcTime,dstTime);
               }

               grad *= sigmoid(value) * spreaderValue * receiverValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable);
               if (!weight.IsKey(propertyIndex)) weight.AddDat(propertyIndex, grad);
               else weight.GetDat(propertyIndex) += grad; 
               /*printf("index:%d, %d,%d: index:%d, sValue:%f, rValue:%f, shapingVal:%f\n",datum.index(),srcNId(),dstNId(),propertyIndex(), \
                                          spreaderValue(), receiverValue(),shapingFunction->Integral(srcTime,dstTime)());*/ 
            }   
         }

      }

      //critical      
      #pragma omp critical
      {
         for (TInt latentVariable=0; latentVariable < topicSize; latentVariable++) {
            THash<TInt,TFlt>& updatedWeight = parameterGrad.kWeights.GetDat(latentVariable);
            THash<TInt,TFlt>& weight = kWeightGradient.GetDat(latentVariable);
            for (THash<TInt,TFlt>::TIter VI = weight.BegI(); !VI.IsEnd(); VI++) {
               if (!updatedWeight.IsKey(VI.GetKey())) updatedWeight.AddDat(VI.GetKey(), VI.GetDat());
               else updatedWeight.GetDat(VI.GetKey()) += VI.GetDat();
            }
         }
      }
   }

   for (TInt latentVariable=0; latentVariable < topicSize; latentVariable++) {
      parameterGrad.kPi.AddDat(latentVariable, latentDistributions.GetDat(datum.index).GetDat(latentVariable));
      parameterGrad.kPi_times.AddDat(latentVariable, 1.0);
   }

   return parameterGrad;
}

void UserPropertyFunction::maximize() {
   for (THash<TInt,TFlt>::TIter PI = parameter.kPi_times.BegI(); !PI.IsEnd(); PI++) {
      PI.GetDat() = 0.0;
   }
}

void UserPropertyFunction::initParameter(Data data, UserPropertyFunctionConfigure configure) {
   allPossibleEdgeNum = 0;
   for (THash<TInt, TCascade>::TIter CI = data.cascH.BegI(); !CI.IsEnd(); CI++) {
      const TCascade& cascade = CI.GetDat();
      TIntPr index;
      for (THash<TInt, THitInfo>::TIter DNI = cascade.BegI(); DNI < cascade.EndI(); DNI++) {
         index.Val2 = DNI.GetKey();
         for (THash<TInt, THitInfo>::TIter SNI = cascade.BegI(); SNI < DNI; SNI++) {
            index.Val1 = SNI.GetKey();
            if (!allPossibleEdges.IsKey(index)) {
               allPossibleEdges.AddDat(index, 1.0);
               allPossibleEdgeNum++;
            }
            else allPossibleEdges.GetDat(index)++;
         }
      } 
   }

   printf("all possible edges: %d\n", allPossibleEdgeNum());

   parameter.init(data); 
   parameterGrad.init(data); 
}

void updateRProp(TFlt initVal, THash<TIntPr,TFlt>& lr, THash<TIntPr,TFlt>& gradient) {
   for(THash<TIntPr,TFlt>::TIter GI = gradient.BegI(); !GI.IsEnd(); GI++) {
      TIntPr key = GI.GetKey();
      if (!lr.IsKey(key)) {
         lr.AddDat(key, initVal);
         if (GI.GetDat() < 0.0) lr.GetDat(key) *= -1.0;
      }
      else {
         if (GI.GetDat() * lr.GetDat(key) > 0.0) lr.GetDat(key) *= 1.2;
         else lr.GetDat(key) *= -0.5;
      }
      GI.GetDat() = lr.GetDat(key);    
   }
}

void UserPropertyFunction::calculateRProp(TFlt initVal, UserPropertyParameter& lr, UserPropertyParameter& gradient) {
   //updateRProp(initVal, lr.acquaintance, gradient.acquaintance);
   //updateRProp(initVal, lr.receiverProperty, gradient.receiverProperty);
   //updateRProp(initVal, lr.spreaderProperty, gradient.spreaderProperty);
   //updateRProp(initVal, lr.topicReceive, gradient.topicReceive);
}

void updateRMSProp(TFlt alpha, THash<TIntPr,TFlt>& lr, THash<TIntPr,TFlt>& gradient) {
   for(THash<TIntPr,TFlt>::TIter GI = gradient.BegI(); !GI.IsEnd(); GI++) {
      TIntPr key = GI.GetKey();
      if (!lr.IsKey(key)) lr.AddDat(key, TMath::Sqrt(GI.GetDat() * GI.GetDat()));
      else lr.GetDat(key) = TMath::Sqrt(alpha * lr.GetDat(key) * lr.GetDat(key) + (1.0 - alpha) * GI.GetDat() * GI.GetDat());
      GI.GetDat() /= lr.GetDat(key);
   }
}

void UserPropertyFunction::calculateRMSProp(TFlt alpha, UserPropertyParameter& lr, UserPropertyParameter& gradient) {
   //updateRMSProp(alpha, lr.acquaintance, gradient.acquaintance);
   //updateRMSProp(alpha, lr.receiverProperty, gradient.receiverProperty);
   //updateRMSProp(alpha, lr.spreaderProperty, gradient.spreaderProperty);
   //updateRMSProp(alpha, lr.topicReceive, gradient.topicReceive);
}

void updateAverageRMSProp(TFlt alpha, TFlt& sigma, THash<TIntPr,TFlt>& gradient) {
   TFlt total = 0.0;
   for(THash<TIntPr,TFlt>::TIter GI = gradient.BegI(); !GI.IsEnd(); GI++) total += GI.GetDat() * GI.GetDat();
   sigma = alpha * sigma * sigma + (1.0 - alpha) * total;
   sigma = TMath::Sqrt(sigma);
   for(THash<TIntPr,TFlt>::TIter GI = gradient.BegI(); !GI.IsEnd(); GI++) GI.GetDat() /= sigma;
}

void UserPropertyFunction::calculateAverageRMSProp(TFlt alpha, TFltV& sigmaes, UserPropertyParameter& gradient) {
   //updateAverageRMSProp(alpha, sigmaes[0], gradient.acquaintance);
   //updateAverageRMSProp(alpha, sigmaes[1], gradient.receiverProperty);
   //updateAverageRMSProp(alpha, sigmaes[2], gradient.spreaderProperty);
   //updateAverageRMSProp(alpha, sigmaes[3], gradient.topicReceive);
}

UserPropertyParameter::UserPropertyParameter() {
   reset();
}

UserPropertyParameter& UserPropertyParameter::operator = (const UserPropertyParameter& p) {
   reset();
   configure = p.configure;
   kWeights = p.kWeights;
   receiverProperty = p.receiverProperty;
   spreaderProperty = p.spreaderProperty;
   kPi = p.kPi;
   kPi_times = p.kPi_times;
   return *this; 
}

/*void UserPropertyParameter::AddEqualTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src) {
   for (THash<TIntPr,TFlt>::TIter PI = src.BegI(); !PI.IsEnd(); PI++) {
      TIntPr key = PI.GetKey();
      TFlt value = PI.GetDat();
      if (!dst.IsKey(key)) dst.AddDat(key,value);
      else dst.GetDat(key) += value;
      //printf("%d,%d: += value:%f, m:%f\n", PI.GetDat(), key.Val1(), key.Val2(), PI.GetDat(), m);
   }
}*/

UserPropertyParameter& UserPropertyParameter::operator += (const UserPropertyParameter& p) {
   for (THash<TInt, THash<TInt,TFlt> >::TIter WI = p.kWeights.BegI(); !WI.IsEnd(); WI++) {
      if (!kWeights.IsKey(WI.GetKey())) kWeights.AddDat(WI.GetKey(), THash<TInt,TFlt>());
      THash<TInt,TFlt>& weight = kWeights.GetDat(WI.GetKey());
      for (THash<TInt,TFlt>::TIter VI = WI.GetDat().BegI(); !VI.IsEnd(); VI++) {
         if (!weight.IsKey(VI.GetKey())) weight.AddDat(VI.GetKey(), VI.GetDat());
         else weight.GetDat(VI.GetKey()) += VI.GetDat();
      }
   }
   
   for(THash<TInt,TFlt>::TIter PI = p.kPi.BegI(); !PI.IsEnd(); PI++) {
      TInt key = PI.GetKey();
      if (!kPi.IsKey(key)) {
         kPi.AddDat(key,0.0);
         kPi_times.AddDat(key,0.0);
      }
      kPi.GetDat(key) += PI.GetDat();
      kPi_times.GetDat(key) += p.kPi_times.GetDat(key);
   }
   return *this; 
}

/*void UserPropertyParameter::MultiplyTHash(THash<TIntPr,TFlt>& dst, const TFlt multiplier) {
   for (THash<TIntPr, TFlt>::TIter VI = dst.BegI(); !VI.IsEnd(); VI++) {
      VI.GetDat() *= multiplier;
   }
}*/

UserPropertyParameter& UserPropertyParameter::operator *= (const TFlt multiplier) {
   for (THash<TInt, THash<TInt,TFlt> >::TIter WI = kWeights.BegI(); !WI.IsEnd(); WI++) {
      for (THash<TInt,TFlt>::TIter VI = WI.GetDat().BegI(); !VI.IsEnd(); VI++) {
         VI.GetDat() *= multiplier;
      }
   }
   return *this; 
}

/*void UserPropertyParameter::UpdateTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src, TFlt minValue, TFlt initValue, TFlt maxValue, TStr comment) {
   for (THash<TIntPr,TFlt>::TIter PI = src.BegI(); !PI.IsEnd(); PI++) {
      TIntPr key = PI.GetKey();
      TFlt valueGradient = PI.GetDat(), value;
      //printf("%s %d,%d: %f, dat:%f, m:%f\n", comment(), key.Val1(), key.Val2(), value(), PI.GetDat()(),m());
      if (dst.IsKey(key)) value = dst.GetDat(key);
      else value = initValue;

      if (comment != TStr("acquaintance"))
         value -= (valueGradient + (configure.Regularizer ? configure.Mu : TFlt(0.0)) * value);
      else
         value -= valueGradient;

      if (value < minValue) value = minValue;
      if (value > maxValue) value = maxValue;

      if (!dst.IsKey(key)) dst.AddDat(key,value);
      else dst.GetDat(key) = value;
      //printf("%s %d,%d: %f, updated\n", comment(), key.Val1(), key.Val2(), dst.GetDat(key)());
   }
}*/

UserPropertyParameter& UserPropertyParameter::projectedlyUpdateGradient(const UserPropertyParameter& p) {
   for (THash<TInt, THash<TInt,TFlt> >::TIter WI = p.kWeights.BegI(); !WI.IsEnd(); WI++) {
      const THash<TInt,TFlt>& weightGradient = WI.GetDat();
      THash<TInt,TFlt>& updatedWeight = kWeights.GetDat(WI.GetKey());
      for (THash<TInt,TFlt>::TIter VI = weightGradient.BegI(); !VI.IsEnd(); VI++) {
         TFlt& value = updatedWeight.GetDat(VI.GetKey());
         value -= (VI.GetDat() + (configure.Regularizer ? configure.Mu : TFlt(0.0)) * value);
      }
   }
   for(THash<TInt,TFlt>::TIter PI = p.kPi.BegI(); !PI.IsEnd(); PI++) {
      TInt key = PI.GetKey();
      TFlt old = kPi.GetDat(key) * kPi_times.GetDat(key);
      kPi_times.GetDat(key) += p.kPi_times.GetDat(key);
      kPi.GetDat(key) = (old + p.kPi.GetDat(key))/kPi_times.GetDat(key);
      //printf("topic %d, prior probability:%f, ", key(), kPi.GetDat(key)());
   }
   //if (!p.kPi.Empty()) printf("\n");
   return *this; 
}

void UserPropertyParameter::reset() {
   kPi.Clr();
   kPi_times.Clr();
   for (TInt topic=0; topic < configure.topicSize; topic++) {
      kWeights.GetDat(topic).Clr();
   }
}

void UserPropertyParameter::set(UserPropertyFunctionConfigure configure) {
  
   this->configure = configure.parameter;
 
   TRnd rnd; rnd.PutSeed(time(NULL));
   for (TInt i=0; i < configure.parameter.topicSize;i++) {
      kPi.AddDat(i,rnd.GetUniDev() * 1.0 + 1.0);
      kPi_times.AddDat(i,0.0);
   }
   TFlt sum = 0.0;
   for (TInt i=0;i<configure.parameter.topicSize;i++) sum += kPi.GetDat(i);
   for (TInt i=0;i<configure.parameter.topicSize;i++) kPi.GetDat(i) /= sum;
}

void UserPropertyParameter::init(Data data) {
   TRnd rnd; rnd.PutSeed(time(NULL));

   for (TInt topic=0; topic < configure.topicSize; topic++) {
      kWeights.AddDat(topic, THash<TInt, TFlt>());
      THash<TInt, TFlt>& weight = kWeights.GetDat(topic);

      for (TInt index=0; index < configure.propertySize; index++) {
         weight.AddDat(index, rnd.GetNrmDev(0.0, 0.001, -1.0, 1.0));
      } 
   }
}

void UserPropertyParameter::GenParameters(TStrFltFltHNEDNet& network) {
   TRnd rnd; rnd.PutSeed(time(NULL));
   for (TStrFltFltHNEDNet::TNodeI NI = network.BegNI(); NI < network.EndNI(); NI++) {
      TIntPr i; i.Val1 = NI.GetId();

      for (TInt index=0; index < configure.propertySize; index++) {
         i.Val2 = index;
         spreaderProperty.AddDat(i,rnd.GetNrmDev(0.0, 2.0, -10.0, 10.0));
         receiverProperty.AddDat(i,rnd.GetNrmDev(0.0, 2.0, -10.0, 10.0));
      }
   }

     
   for (TInt topic=0; topic < configure.topicSize; topic++) {
      kWeights.AddDat(topic, THash<TInt, TFlt>());
      THash<TInt, TFlt>& weight = kWeights.GetDat(topic);

      for (TInt index=0; index < configure.propertySize; index++) {
         weight.AddDat(index, rnd.GetNrmDev(0.0, 2.0, -10.0, 10.0));
      } 
   }
  
}

TFlt UserPropertyParameter::GetValue(TInt srcNId, TInt dstNId, TInt topic) const {
   TFlt value = 0.0;
   TIntPr srcIndex(srcNId,0), dstIndex(dstNId,0);
   const THash<TInt,TFlt>& weight = kWeights.GetDat(topic);  

   for (TInt index=0; index < configure.propertySize; index++) {
      srcIndex.Val2 = dstIndex.Val2 = index;
      value += spreaderProperty.GetDat(srcIndex) * receiverProperty.GetDat(dstIndex) * weight.GetDat(index);
   }
   return value;
}

TFlt UserPropertyParameter::GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const {
   TFlt value = GetValue(srcNId, dstNId, topic);
   TFlt alpha = ReLU(value);
   if (alpha > configure.MaxAlpha) alpha = configure.MaxAlpha;
   else if (alpha < 0.0001) alpha = 0.0001;
   return alpha;
}
