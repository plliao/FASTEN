#include <UserPropertyFunction.h>
#include <ctime>

TFlt sigmoid(TFlt t) {
   return 1.0/(1.0 + TMath::Power(TMath::E,-1.0*t));
}

void UserPropertyFunction::set(UserPropertyFunctionConfigure configure) {
   shapingFunction = configure.shapingFunction;
   latentVariableSize = configure.topicSize;
   propertySize = configure.propertySize;
   MaxAlpha = configure.MaxAlpha;
   MinAlpha = configure.MinAlpha;
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
                        
         TFlt alpha = 0.0;
         TIntPr spreaderIndex, receiverIndex, acquaintanceIndex;
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         acquaintanceIndex.Val1 = srcNId; acquaintanceIndex.Val1 = dstNId;
         for (int propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
            receiverIndex.Val2 = spreaderIndex.Val2 = propertyIndex;
            TFlt receiverValue = parameter.propertyInitValue, spreaderValue = parameter.propertyInitValue;
            if (parameter.receiverProperty.IsKey(receiverIndex)) receiverValue = parameter.receiverProperty.GetDat(receiverIndex);
            if (parameter.spreaderProperty.IsKey(spreaderIndex)) spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);
            alpha += receiverValue * spreaderValue;
         }
         alpha /= (TFlt)propertySize;
         
         receiverIndex.Val2 = spreaderIndex.Val2 = latentVariable; 
         TFlt receiverValue = parameter.topicInitValue, spreaderValue = parameter.topicInitValue;
         if (parameter.topicReceive.IsKey(receiverIndex)) receiverValue = parameter.topicReceive.GetDat(receiverIndex);
         if (parameter.topicSpread.IsKey(spreaderIndex)) spreaderValue = parameter.topicSpread.GetDat(spreaderIndex);
         alpha += receiverValue * spreaderValue;
         
         TFlt acquaintanceValue = parameter.acquaintanceInitValue;
         if (parameter.acquaintance.IsKey(acquaintanceIndex)) acquaintanceValue = parameter.acquaintance.GetDat(acquaintanceIndex);
         alpha = acquaintanceValue * MaxAlpha * sigmoid(alpha);

         //if (alpha < 0.0005) alpha = 0.0005;
         //else if (alpha > MaxAlpha) alpha = MaxAlpha;
 
         sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
         val += alpha * shapingFunction->Integral(srcTime,dstTime);
         //printf("datum:%d, topic:%d, sumInLog:%f, val:%f, alpha:%f, shapingVal:%f, shapingInt:%f\n", \
                datum.index(), latentVariable(), sumInLog(),val(),alpha(),shapingFunction->Value(srcTime,dstTime)(),shapingFunction->Integral(srcTime,dstTime)()); 
      }
      lossTable[i] = val;
      
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime && sumInLog!=0.0) lossTable[i] -= TMath::Log(sumInLog);
      
   }

   for (int i=0;i<nodeSize;i++) totalLoss += lossTable[i];
   delete[] lossTable;

   TFlt logP = -1 * totalLoss;
   TFlt logPi = TMath::Log(parameter.kPi.GetDat(latentVariable));
   printf("datum:%d, topic:%d, logP: %f, logPi=%f\n",datum.index(), latentVariable(), logP(), logPi());
   return logP + logPi;
}

UserPropertyParameter& UserPropertyFunction::gradient(Datum datum) {
   return parameterGrad;
}

UserPropertyParameter& UserPropertyFunction::gradient1(Datum datum) {
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

      TIntPr acquaintanceIndex; acquaintanceIndex.Val2 = dstNId;
      TIntPr receiverIndex, spreaderIndex;
      
      bool inCascade = false;
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) {
         dstTime = Cascade.GetTm(dstNId);
         inCascade = true;
      }
      else dstTime = CurrentTime;

      THash<TIntPr,TFlt> propertyValueVector;
      THash<TInt,TFlt> dstAlphaVector;
      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
         
         acquaintanceIndex.Val1 = srcNId;
         TFlt acquaintedValue = parameter.acquaintanceInitValue;               
         if (parameter.acquaintance.IsKey(acquaintanceIndex)) acquaintedValue = parameter.acquaintance.GetDat(acquaintanceIndex);
            
         TFlt alpha = 0.0;
         receiverIndex.Val1 = dstNId; spreaderIndex.Val1 = srcNId;
         for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
            receiverIndex.Val2 = spreaderIndex.Val2 = propertyIndex;
            TFlt spreaderValue = parameter.propertyInitValue, receiverValue = parameter.propertyInitValue;
            if (parameter.spreaderProperty.IsKey(spreaderIndex)) spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);
            if (parameter.receiverProperty.IsKey(receiverIndex)) receiverValue = parameter.receiverProperty.GetDat(receiverIndex);
            alpha += spreaderValue * receiverValue;
         }
         alpha /= (TFlt)propertySize;         

         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            receiverIndex.Val2 = spreaderIndex.Val2 = latentVariable; 
            TFlt spreaderValue = parameter.topicInitValue, receiverValue = parameter.topicInitValue;
            if (parameter.topicSpread.IsKey(spreaderIndex)) spreaderValue = parameter.topicSpread.GetDat(spreaderIndex);
            if (parameter.topicReceive.IsKey(receiverIndex)) receiverValue = parameter.topicReceive.GetDat(receiverIndex);

            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
            propertyValueVector.AddDat(propertyValueIndex, alpha + spreaderValue * receiverValue);

            if (inCascade) {
               TFlt hazard = acquaintedValue * MaxAlpha * sigmoid(propertyValueVector.GetDat(propertyValueIndex)) * shapingFunction->Value(srcTime,dstTime);
               if (!dstAlphaVector.IsKey(latentVariable)) dstAlphaVector.AddDat(latentVariable,hazard);
               else dstAlphaVector.GetDat(latentVariable) += hazard;
            }
         }
      }

      THash<TIntPr,TFlt> receiverPropertyGrad(propertySize());
      THash<TIntPr,TFlt> topicReceiveGrad(latentVariableSize());

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
            
         acquaintanceIndex.Val1 = srcNId;

         TFlt acquaintedValue = parameter.acquaintanceInitValue;               
         if (parameter.acquaintance.IsKey(acquaintanceIndex)) acquaintedValue = parameter.acquaintance.GetDat(acquaintanceIndex);
                        
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
            spreaderIndex.Val2 = receiverIndex.Val2 = propertyIndex;
            TFlt spreaderValue = parameter.propertyInitValue;
            if (parameter.spreaderProperty.IsKey(spreaderIndex)) spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);

            for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
               TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
               TFlt propertyValue = sigmoid(propertyValueVector.GetDat(propertyValueIndex));
               propertyValue = MaxAlpha * propertyValue * (1.0 - propertyValue);
               
               TFlt grad;
               if (inCascade) {
                  TFlt totalAlpha = dstAlphaVector.GetDat(latentVariable);
                  grad = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/totalAlpha;
               }
               else grad = shapingFunction->Integral(srcTime,dstTime);
               grad *= acquaintedValue * propertyValue * spreaderValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable)/ (TFlt)propertySize;
            
               if (!receiverPropertyGrad.IsKey(receiverIndex)) receiverPropertyGrad.AddDat(receiverIndex,grad);
               else receiverPropertyGrad.GetDat(receiverIndex) += grad;
            }
            //printf("index:%d, %d,%d: index:%d, sValue:%f, rValue:%f, shapingVal:%f\n",datum.index(),srcNId(),dstNId(),propertyIndex(),sValue(),rValue(),shapingFunction->Integral(srcTime,dstTime)()); 
         }
         
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            spreaderIndex.Val2 = receiverIndex.Val2 = latentVariable;
            TFlt spreaderValue = parameter.topicInitValue;
            if (parameter.topicSpread.IsKey(spreaderIndex)) spreaderValue = parameter.topicSpread.GetDat(spreaderIndex);
               
            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
            TFlt propertyValue = sigmoid(propertyValueVector.GetDat(propertyValueIndex));
            propertyValue = MaxAlpha * propertyValue * (1.0 - propertyValue);
               
            TFlt grad;
            if (inCascade) {
               TFlt totalAlpha = dstAlphaVector.GetDat(latentVariable);
               grad = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/totalAlpha;
            }
            else grad = shapingFunction->Integral(srcTime,dstTime);
            grad *= acquaintedValue * propertyValue * spreaderValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable);
            
            if (!topicReceiveGrad.IsKey(receiverIndex)) topicReceiveGrad.AddDat(receiverIndex,grad);
            else topicReceiveGrad.GetDat(receiverIndex) += grad;
         }
      }

      //critical      
      #pragma omp critical
      {
         for (THash<TIntPr,TFlt>::TIter I = receiverPropertyGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.receiverProperty.IsKey(I.GetKey())) parameterGrad.receiverProperty.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.receiverProperty.GetDat(I.GetKey()) += I.GetDat();
            //printf("%d,%d receiver property grad:%f\n",I.GetKey().Val1(),I.GetKey().Val2(),I.GetDat()());
         } 
         for (THash<TIntPr,TFlt>::TIter I = topicReceiveGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.topicReceive.IsKey(I.GetKey())) parameterGrad.topicReceive.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.topicReceive.GetDat(I.GetKey()) += I.GetDat();
            //printf("%d,%d topic receive grad:%f\n",I.GetKey().Val1(),I.GetKey().Val2(),I.GetDat()());
         } 
      }
   }

   /*for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
      parameterGrad.kPi.GetDat(latentVariable) = latentDistributions.GetDat(datum.index).GetDat(latentVariable);
      parameterGrad.kPi_times.GetDat(latentVariable)++;
   }*/

   return parameterGrad;
}

UserPropertyParameter& UserPropertyFunction::gradient2(Datum datum) {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;

   parameterGrad.reset();
   
   int nodeSize = NodeNmH.Len();
   int cascadeSize = Cascade.Len();

   #pragma omp parallel for
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      TInt dstNId = key, srcNId;
      TFlt dstTime, srcTime;

      TIntPr acquaintanceIndex; acquaintanceIndex.Val2 = dstNId;
      TIntPr receiverIndex, spreaderIndex;
      
      bool inCascade = false;
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) {
         dstTime = Cascade.GetTm(dstNId);
         inCascade = true;
      }
      else dstTime = CurrentTime;

      THash<TIntPr,TFlt> propertyValueVector;
      THash<TInt,TFlt> dstAlphaVector;
      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
         
         acquaintanceIndex.Val1 = srcNId;
         TFlt acquaintedValue = parameter.acquaintanceInitValue;               
         if (parameter.acquaintance.IsKey(acquaintanceIndex)) acquaintedValue = parameter.acquaintance.GetDat(acquaintanceIndex);
            
         TFlt alpha = 0.0;
         receiverIndex.Val1 = dstNId; spreaderIndex.Val1 = srcNId;
         for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
            receiverIndex.Val2 = spreaderIndex.Val2 = propertyIndex;
            TFlt spreaderValue = parameter.propertyInitValue, receiverValue = parameter.propertyInitValue;
            if (parameter.spreaderProperty.IsKey(spreaderIndex)) spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);
            if (parameter.receiverProperty.IsKey(receiverIndex)) receiverValue = parameter.receiverProperty.GetDat(receiverIndex);
            alpha += spreaderValue * receiverValue;
         }
         alpha /= (TFlt)propertySize;         

         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            receiverIndex.Val2 = spreaderIndex.Val2 = latentVariable; 
            TFlt spreaderValue = parameter.topicInitValue, receiverValue = parameter.topicInitValue;
            if (parameter.topicSpread.IsKey(spreaderIndex)) spreaderValue = parameter.topicSpread.GetDat(spreaderIndex);
            if (parameter.topicReceive.IsKey(receiverIndex)) receiverValue = parameter.topicReceive.GetDat(receiverIndex);

            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
            propertyValueVector.AddDat(propertyValueIndex, alpha + spreaderValue * receiverValue);

            if (inCascade) {
               TFlt hazard = acquaintedValue * MaxAlpha * sigmoid(propertyValueVector.GetDat(propertyValueIndex)) * shapingFunction->Value(srcTime,dstTime);
               if (!dstAlphaVector.IsKey(latentVariable)) dstAlphaVector.AddDat(latentVariable,hazard);
               else dstAlphaVector.GetDat(latentVariable) += hazard;
            }
         }
      }
      
      THash<TIntPr,TFlt> spreaderPropertyGrad(cascadeSize * propertySize());
      THash<TIntPr,TFlt> topicSpreadGrad(cascadeSize * latentVariableSize());

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
         
         acquaintanceIndex.Val1 = srcNId;

         TFlt acquaintedValue = parameter.acquaintanceInitValue;               
         if (parameter.acquaintance.IsKey(acquaintanceIndex)) acquaintedValue = parameter.acquaintance.GetDat(acquaintanceIndex);
                        
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
            spreaderIndex.Val2 = receiverIndex.Val2 = propertyIndex;
            TFlt receiverValue = parameter.propertyInitValue;
            if (parameter.receiverProperty.IsKey(receiverIndex)) receiverValue = parameter.receiverProperty.GetDat(receiverIndex);

            for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
               TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
               TFlt propertyValue = sigmoid(propertyValueVector.GetDat(propertyValueIndex));
               propertyValue = MaxAlpha * propertyValue * (1.0 - propertyValue);
               
               TFlt grad;
               if (inCascade) {
                  TFlt totalAlpha = dstAlphaVector.GetDat(latentVariable);
                  grad = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/totalAlpha;
               }
               else grad = shapingFunction->Integral(srcTime,dstTime);
               grad *= acquaintedValue * propertyValue * receiverValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable)/ (TFlt)propertySize;
            
               if (!spreaderPropertyGrad.IsKey(spreaderIndex)) spreaderPropertyGrad.AddDat(spreaderIndex,grad);
               else spreaderPropertyGrad.GetDat(spreaderIndex) += grad;
            }

            //printf("index:%d, %d,%d: index:%d, sValue:%f, rValue:%f, shapingVal:%f\n",datum.index(),srcNId(),dstNId(),propertyIndex(),sValue(),rValue(),shapingFunction->Integral(srcTime,dstTime)()); 
         }
         
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            spreaderIndex.Val2 = receiverIndex.Val2 = latentVariable;
            TFlt receiverValue = parameter.topicInitValue;
            if (parameter.topicReceive.IsKey(receiverIndex)) receiverValue = parameter.topicReceive.GetDat(receiverIndex);
               
            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
            TFlt propertyValue = sigmoid(propertyValueVector.GetDat(propertyValueIndex));
            propertyValue = MaxAlpha * propertyValue * (1.0 - propertyValue);
               
            TFlt grad;
            if (inCascade) {
               TFlt totalAlpha = dstAlphaVector.GetDat(latentVariable);
               grad = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/totalAlpha;
            }
            else grad = shapingFunction->Integral(srcTime,dstTime);
            grad *= acquaintedValue * propertyValue * receiverValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable);
            
            if (!topicSpreadGrad.IsKey(spreaderIndex)) topicSpreadGrad.AddDat(spreaderIndex,grad);
            else topicSpreadGrad.GetDat(spreaderIndex) += grad;
         }
      }
      //critical      
      #pragma omp critical
      {
         for (THash<TIntPr,TFlt>::TIter I = spreaderPropertyGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.spreaderProperty.IsKey(I.GetKey())) parameterGrad.spreaderProperty.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.spreaderProperty.GetDat(I.GetKey()) += I.GetDat();
            //printf("%d,%d spreader property grad:%f\n",I.GetKey().Val1(),I.GetKey().Val2(),I.GetDat()());
         } 
         for (THash<TIntPr,TFlt>::TIter I = topicSpreadGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.topicSpread.IsKey(I.GetKey())) parameterGrad.topicSpread.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.topicSpread.GetDat(I.GetKey()) += I.GetDat();
         }
      }
   }

   for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
      parameterGrad.kPi.AddDat(latentVariable, latentDistributions.GetDat(datum.index).GetDat(latentVariable));
      parameterGrad.kPi_times.AddDat(latentVariable,1.0);
   }

   return parameterGrad;
}

UserPropertyParameter& UserPropertyFunction::gradient3(Datum datum) {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;

   parameterGrad.reset();
   
   int nodeSize = NodeNmH.Len();

   #pragma omp parallel for
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      TInt dstNId = key, srcNId;
      TFlt propertyValue = 0.0, propertyVal = 0.0;
      TFlt dstTime, srcTime;

      TIntPr acquaintanceIndex; acquaintanceIndex.Val2 = dstNId;
      TIntPr receiverIndex, spreaderIndex;
      
      bool inCascade = false;
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) {
         dstTime = Cascade.GetTm(dstNId);
         inCascade = true;
      }
      else dstTime = CurrentTime;

      THash<TIntPr,TFlt> propertyValueVector;
      THash<TInt,TFlt> dstAlphaVector;
      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
         
         acquaintanceIndex.Val1 = srcNId;
         TFlt acquaintedValue = parameter.acquaintanceInitValue;               
         if (parameter.acquaintance.IsKey(acquaintanceIndex)) acquaintedValue = parameter.acquaintance.GetDat(acquaintanceIndex);
            
         TFlt alpha = 0.0;
         receiverIndex.Val1 = dstNId; spreaderIndex.Val1 = srcNId;
         for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
            receiverIndex.Val2 = spreaderIndex.Val2 = propertyIndex;
            TFlt spreaderValue = parameter.propertyInitValue, receiverValue = parameter.propertyInitValue;
            if (parameter.spreaderProperty.IsKey(spreaderIndex)) spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);
            if (parameter.receiverProperty.IsKey(receiverIndex)) receiverValue = parameter.receiverProperty.GetDat(receiverIndex);
            alpha += spreaderValue * receiverValue;
         }
         alpha /= (TFlt)propertySize;         

         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            receiverIndex.Val2 = spreaderIndex.Val2 = latentVariable; 
            TFlt spreaderValue = parameter.topicInitValue, receiverValue = parameter.topicInitValue;
            if (parameter.topicSpread.IsKey(spreaderIndex)) spreaderValue = parameter.topicSpread.GetDat(spreaderIndex);
            if (parameter.topicReceive.IsKey(receiverIndex)) receiverValue = parameter.topicReceive.GetDat(receiverIndex);

            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
            propertyValueVector.AddDat(propertyValueIndex, alpha + spreaderValue * receiverValue);

            if (inCascade) {
               TFlt hazard = acquaintedValue * MaxAlpha * sigmoid(propertyValueVector.GetDat(propertyValueIndex)) * shapingFunction->Value(srcTime,dstTime);
               if (!dstAlphaVector.IsKey(latentVariable)) dstAlphaVector.AddDat(latentVariable,hazard);
               else dstAlphaVector.GetDat(latentVariable) += hazard;
            }
         }
      }

      THash<TIntPr,TFlt> acquaintanceGrad;

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break;

         acquaintanceIndex.Val1 = srcNId; 
            
         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;           
            TFlt propertyValue = MaxAlpha * sigmoid(propertyValueVector.GetDat(propertyValueIndex)); 
            TFlt grad;

            if (inCascade) {
               TFlt topicTotalAlpha = dstAlphaVector.GetDat(latentVariable);
               grad = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime) / topicTotalAlpha;
               //printf("%d,%d, acquaintance grad:%f, topicTotalAlpha:%f, topic:%d\n",srcNId(),dstNId(),grad(),topicTotalAlpha());
            }
            else
               grad = shapingFunction->Integral(srcTime,dstTime);
            grad *= propertyValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable);
            //printf("%d,%d, acquaintance grad:%f, topic:%d, topic probability:%f\n",srcNId(),dstNId(),grad(),latentVariable(),latentDistributions.GetDat(datum.index).GetDat(latentVariable)());

            if (!acquaintanceGrad.IsKey(acquaintanceIndex)) acquaintanceGrad.AddDat(acquaintanceIndex,grad);
            else acquaintanceGrad.GetDat(acquaintanceIndex) += grad;
         }
         //printf("%d,%d, acquaintance grad:%f, srcTime:%f, dstTime:%f\n",srcNId(),dstNId(),acquaintanceGrad.GetDat(acquaintanceIndex)(), srcTime(), dstTime());
      }

      //critical      
      #pragma omp critical
      {
         for (THash<TIntPr,TFlt>::TIter I = acquaintanceGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.acquaintance.IsKey(I.GetKey())) parameterGrad.acquaintance.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.acquaintance.GetDat(I.GetKey()) += I.GetDat();
            //printf("%d,%d acquaintance grad:%f\n",I.GetKey().Val1(),I.GetKey().Val2(),I.GetDat()());
         } 
      }
   }

   /*for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
      parameterGrad.kPi.GetDat(latentVariable) = latentDistributions.GetDat(datum.index).GetDat(latentVariable);
      parameterGrad.kPi_times.GetDat(latentVariable)++;
   }*/

   return parameterGrad;
}

void UserPropertyFunction::maximize() {
   for (THash<TInt,TFlt>::TIter PI = parameter.kPi_times.BegI(); !PI.IsEnd(); PI++) {
      PI.GetDat() = 0.0;
   }
}

UserPropertyParameter::UserPropertyParameter() {
   reset();
}

UserPropertyParameter& UserPropertyParameter::operator = (const UserPropertyParameter& p) {
   reset();
   multiplier = p.multiplier;
   propertyInitValue = p.propertyInitValue;
   propertyMaxValue = p.propertyMaxValue;
   propertyMinValue = p.propertyMinValue;
   topicInitValue = p.topicInitValue;
   topicMaxValue = p.topicMaxValue;
   topicMinValue = p.topicMinValue;
   acquaintanceInitValue = p.acquaintanceInitValue;
   acquaintanceMaxValue = p.acquaintanceMaxValue;
   acquaintanceMinValue = p.acquaintanceMinValue;
   acquaintance = p.acquaintance;
   receiverProperty = p.receiverProperty;
   spreaderProperty = p.spreaderProperty;
   topicReceive = p.topicReceive;
   topicSpread = p.topicSpread;
   kPi = p.kPi;
   kPi_times = p.kPi_times;
   return *this; 
}

void UserPropertyParameter::AddEqualTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src, TFlt m) {
   for (THash<TIntPr,TFlt>::TIter PI = src.BegI(); !PI.IsEnd(); PI++) {
      TIntPr key = PI.GetKey();
      TFlt value = PI.GetDat() * m;
      if (!dst.IsKey(key)) dst.AddDat(key,value/multiplier);
      else dst.GetDat(key) += value/multiplier;
      //printf("%d,%d: += value:%f, m:%f\n", PI.GetDat(), key.Val1(), key.Val2(), PI.GetDat(), m);
   }
}

UserPropertyParameter& UserPropertyParameter::operator += (const UserPropertyParameter& p) {
   AddEqualTHash(acquaintance, p.acquaintance, p.multiplier);
   AddEqualTHash(receiverProperty, p.receiverProperty, p.multiplier);
   AddEqualTHash(spreaderProperty, p.spreaderProperty, p.multiplier);
   AddEqualTHash(topicReceive, p.topicReceive, p.multiplier);
   AddEqualTHash(topicSpread, p.topicSpread, p.multiplier);
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

UserPropertyParameter& UserPropertyParameter::operator *= (const TFlt multiplier) {
   this->multiplier *= multiplier;
   return *this; 
}

void UserPropertyParameter::UpdateTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src, TFlt m, TFlt minValue, TFlt initValue, TFlt maxValue, TStr comment) {
   for (THash<TIntPr,TFlt>::TIter PI = src.BegI(); !PI.IsEnd(); PI++) {
      TIntPr key = PI.GetKey();
      TFlt value = PI.GetDat() * m;
      //printf("%s %d,%d: %f, dat:%f, m:%f\n", comment(), key.Val1(), key.Val2(), value(), PI.GetDat()(),m());
      if (dst.IsKey(key)) value = dst.GetDat(key) * multiplier - value;
      else value = initValue - value;

      if (value < minValue) value = minValue;
      if (value > maxValue) value = maxValue;

      if (!dst.IsKey(key)) dst.AddDat(key,value/multiplier);
      else dst.GetDat(key) = value/multiplier;
      //printf("%s %d,%d: %f, updated\n", comment(), key.Val1(), key.Val2(), dst.GetDat(key)());
   }
}

UserPropertyParameter& UserPropertyParameter::projectedlyUpdateGradient(const UserPropertyParameter& p) {
   UpdateTHash(acquaintance, p.acquaintance, p.multiplier, acquaintanceMinValue, acquaintanceInitValue, acquaintanceMaxValue, "acquaintance");
   UpdateTHash(receiverProperty, p.receiverProperty, p.multiplier, -1.0 * propertyMaxValue, propertyInitValue, propertyMaxValue, "receiver property");
   UpdateTHash(spreaderProperty, p.spreaderProperty, p.multiplier, propertyMinValue, propertyInitValue, propertyMaxValue, "spreader property");
   UpdateTHash(topicReceive, p.topicReceive, p.multiplier, -1.0 * topicMaxValue, topicInitValue, topicMaxValue, "topic receiver");
   UpdateTHash(topicSpread, p.topicSpread, p.multiplier, topicMinValue, topicInitValue, topicMaxValue, "topic spreader");
   for(THash<TInt,TFlt>::TIter PI = p.kPi.BegI(); !PI.IsEnd(); PI++) {
      TInt key = PI.GetKey();
      TFlt old = kPi.GetDat(key) * kPi_times.GetDat(key);
      kPi_times.GetDat(key) += p.kPi_times.GetDat(key);
      kPi.GetDat(key) = (old + p.kPi.GetDat(key))/kPi_times.GetDat(key);
      printf("topic %d, prior probability:%f, ", key(), kPi.GetDat(key)());
   }
   if (!p.kPi.Empty()) printf("\n");
   return *this; 
}

void UserPropertyParameter::reset() {
   multiplier = 1.0;
   acquaintance.Clr();
   receiverProperty.Clr();
   spreaderProperty.Clr();
   topicReceive.Clr();
   topicSpread.Clr();
   kPi.Clr();
   kPi_times.Clr();
   /*for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) { 
      piI.GetDat() = 0.0;
      kPi_times.GetDat(piI.GetKey()) = 0.0;
   }*/
}

void UserPropertyParameter::set(UserPropertyFunctionConfigure configure) {
   
   propertyInitValue = configure.propertyInitValue;
   propertyMaxValue = configure.propertyMaxValue;
   propertyMinValue = configure.propertyMinValue;
   
   topicInitValue = configure.topicInitValue;
   topicMaxValue = configure.topicMaxValue;
   topicMinValue = configure.topicMinValue;
   
   acquaintanceInitValue = configure.acquaintanceInitValue;
   acquaintanceMaxValue = configure.acquaintanceMaxValue;
   acquaintanceMinValue = configure.acquaintanceMinValue;
   
   multiplier = 1.0;
   TRnd rnd; rnd.PutSeed(time(NULL));
   for (TInt i=0;i<configure.topicSize;i++) {
      kPi.AddDat(i,1.0);
      kPi_times.AddDat(i,0.0);
   }
   TFlt sum = 0.0;
   for (TInt i=0;i<configure.topicSize;i++) sum += kPi.GetDat(i);
   for (TInt i=0;i<configure.topicSize;i++) kPi.GetDat(i) /= sum;
}

void UserPropertyParameter::init(Data data, UserPropertyFunctionConfigure configure) {
  THash<TInt, TNodeInfo> &NodeNmH = data.NodeNmH;
  TRnd rnd; rnd.PutSeed(time(NULL));
  for (THash<TInt, TNodeInfo>::TIter NI = NodeNmH.BegI(); !NI.IsEnd(); NI++) {
     for (TInt index=0; index<configure.propertySize; index++) {
        TIntPr i; i.Val1 = NI.GetKey(); i.Val2 = index;
        //receiverProperty.AddDat(i,-1.0 * configure.propertyMaxValue() + rnd.GetUniDev() * 2.0 * configure.propertyMaxValue());
        //spreaderProperty.AddDat(i,configure.propertyMinValue() + rnd.GetUniDev() * (configure.propertyMaxValue()-configure.propertyMinValue()));
        //spreaderProperty.AddDat(i,configure.propertyInitValue());
        //receiverProperty.AddDat(i,configure.propertyInitValue());
        spreaderProperty.AddDat(i,rnd.GetNrmDev(configure.propertyInitValue(), 0.001, configure.propertyMinValue(), configure.propertyMaxValue()));
        receiverProperty.AddDat(i,rnd.GetNrmDev(configure.propertyInitValue(), 0.001, configure.propertyMinValue(), configure.propertyMaxValue()));
        //receiverProperty.AddDat(i,rnd.GetNrmDev(0.0, 1.0, -1.0 * configure.propertyMaxValue(), configure.propertyMaxValue()));
        //printf("%d,%d receiverProperty:%f, spreaderProperty:%f\n",i.Val1(),i.Val2(),receiverProperty.GetDat(i)(),spreaderProperty.GetDat(i)());
     }

     for (TInt topic=0; topic<configure.topicSize; topic++) {
        TIntPr i; i.Val1 = NI.GetKey(); i.Val2 = topic;
        //topicReceive.AddDat(i,-1.0 * configure.propertyMaxValue() + rnd.GetUniDev() * 2.0 * configure.propertyMaxValue());
        //topicSpread.AddDat(i,configure.topicMinValue() + rnd.GetUniDev() * (configure.topicMaxValue()-configure.topicMinValue()));
        //topicSpread.AddDat(i,configure.topicInitValue());
        //topicReceive.AddDat(i,configure.topicInitValue());
        topicSpread.AddDat(i,rnd.GetNrmDev(configure.topicInitValue(), 0.001, configure.topicMinValue(), configure.topicMaxValue()));
        topicReceive.AddDat(i,rnd.GetNrmDev(configure.topicInitValue(), 0.001, configure.topicMinValue(), configure.topicMaxValue()));
        //topicReceive.AddDat(i,rnd.GetNrmDev(0.0, 1.0, -1.0 * configure.topicMaxValue(), configure.topicMaxValue()));
        //printf("%d,%d topicReceive:%f, topicSpread:%f\n",i.Val1(),i.Val2(),topicReceive.GetDat(i)(),topicSpread.GetDat(i)());
     }
  } 
}

void UserPropertyParameter::GenParameters(TStrFltFltHNEDNet& network, UserPropertyFunctionConfigure configure) {
   set(configure);
   TRnd rnd; rnd.PutSeed(time(NULL));
   for (TStrFltFltHNEDNet::TNodeI NI = network.BegNI(); NI < network.EndNI(); NI++) {
      TIntPr i; i.Val1 = NI.GetId();

      for (TInt index=0; index<configure.propertySize; index++) {
         i.Val2 = index;
         //receiverProperty.AddDat(i,-1.0 * configure.propertyMaxValue() + rnd.GetUniDev() * 2.0 * configure.propertyMaxValue());
         spreaderProperty.AddDat(i,configure.propertyMinValue() + rnd.GetUniDev() * (configure.propertyMaxValue()-configure.propertyMinValue()));
         receiverProperty.AddDat(i,rnd.GetNrmDev(0.0, 0.1, -1.0 * configure.propertyMaxValue(), configure.propertyMaxValue()));
      }
     
      for (TInt topic=0; topic<configure.topicSize; topic++) {
         i.Val2 = topic;
         //topicReceive.AddDat(i,-1.0 * configure.propertyMaxValue() + rnd.GetUniDev() * 2.0 * configure.propertyMaxValue());
         topicSpread.AddDat(i,configure.topicMinValue() + rnd.GetUniDev() * (configure.topicMaxValue()-configure.topicMinValue()));
         topicReceive.AddDat(i,rnd.GetNrmDev(0.0, 0.1, -1.0 * configure.topicMaxValue(), configure.topicMaxValue()));
      }
   }

   TFlt range = acquaintanceMaxValue - acquaintanceMinValue;
   for (TStrFltFltHNEDNet::TEdgeI EI = network.BegEI(); EI < network.EndEI(); EI++) {
      TIntPr i; i.Val1 = EI.GetSrcNId(); i.Val2 = EI.GetDstNId();
      acquaintance.AddDat(i,rnd.GetUniDev() * range + acquaintanceMinValue);
   }
}

TFlt UserPropertyFunction::GetAcquaitance(TInt srcNId, TInt dstNId) const {
   TIntPr acquaintanceIndex;
   acquaintanceIndex.Val1 = srcNId; acquaintanceIndex.Val2 = dstNId;
   TFlt acquaintedValue = parameter.acquaintanceInitValue;               
   if (parameter.acquaintance.IsKey(acquaintanceIndex)) acquaintedValue = parameter.acquaintance.GetDat(acquaintanceIndex);
   return acquaintedValue; 
}

TFlt UserPropertyFunction::GetPropertyValue(TInt srcNId, TInt dstNId) const {
   TFlt alpha = 0.0;
   TIntPr receiverIndex, spreaderIndex;
   receiverIndex.Val1 = dstNId; spreaderIndex.Val1 = srcNId;
   
   for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
      receiverIndex.Val2 = spreaderIndex.Val2 = propertyIndex;
      TFlt spreaderValue = parameter.propertyInitValue, receiverValue = parameter.propertyInitValue;
      if (parameter.spreaderProperty.IsKey(spreaderIndex)) spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);
      if (parameter.receiverProperty.IsKey(receiverIndex)) receiverValue = parameter.receiverProperty.GetDat(receiverIndex);
         alpha += spreaderValue * receiverValue;
   }
   alpha /= (TFlt)propertySize;
   return alpha;
}

TFlt UserPropertyFunction::GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const {
   TFlt alpha = GetPropertyValue(srcNId,dstNId);
   TFlt acquaintedValue = GetAcquaitance(srcNId,dstNId); 
         
   TIntPr receiverIndex, spreaderIndex;
   receiverIndex.Val1 = dstNId; spreaderIndex.Val1 = srcNId;
   receiverIndex.Val2 = spreaderIndex.Val2 = topic; 
   TFlt spreaderValue = parameter.topicInitValue, receiverValue = parameter.topicInitValue;
   if (parameter.topicSpread.IsKey(spreaderIndex)) spreaderValue = parameter.topicSpread.GetDat(spreaderIndex);
   if (parameter.topicReceive.IsKey(receiverIndex)) receiverValue = parameter.topicReceive.GetDat(receiverIndex);
            
   alpha += spreaderValue * receiverValue;
   alpha = acquaintedValue * MaxAlpha * sigmoid(alpha);
   if (alpha > MaxAlpha) alpha = MaxAlpha;
   else if (alpha < MinAlpha) alpha = MinAlpha;
   return alpha;
}
