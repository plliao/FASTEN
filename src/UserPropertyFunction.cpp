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

         if (GetAcquaitance(srcNId,dstNId)!=1.0) continue;
         if (!shapingFunction->Before(srcTime,dstTime)) break; 
                        
         TFlt alpha = GetAlpha(srcNId, dstNId, latentVariable);
 
         sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
         val += alpha * shapingFunction->Integral(srcTime,dstTime);
         /*printf("datum:%d, topic:%d, sumInLog:%f, val:%f, alpha:%f, shapingVal:%f, shapingInt:%f\n", \
                datum.index(), latentVariable(), sumInLog(),val(),alpha(),shapingFunction->Value(srcTime,dstTime)(),shapingFunction->Integral(srcTime,dstTime)()); */
      }
      lossTable[i] = val;
      
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime && sumInLog != 0.0) lossTable[i] -= TMath::Log(sumInLog);
      //if (lossTable[i] < -10.0) lossTable[i] = -10.0;   
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
   TInt cascadeSize = Cascade.Len();

   #pragma omp parallel for
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      TInt dstNId = key, srcNId;
      TFlt dstTime, srcTime;

      //TIntPr acquaintanceIndex; acquaintanceIndex.Val2 = dstNId;
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

         if (GetAcquaitance(srcNId,dstNId)!=1.0) continue;
         if (!shapingFunction->Before(srcTime,dstTime)) break; 
         
         //TFlt acquaintedValue = GetAcquaitance(srcNId, dstNId);             
         TFlt propertyValue = GetPropertyValue(srcNId, dstNId);

         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            TFlt topicValue = GetTopicValue(srcNId, dstNId, latentVariable);
            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
            propertyValueVector.AddDat(propertyValueIndex, propertyValue + topicValue);

            if (inCascade) {
               TFlt hazard = ReLU(propertyValue + topicValue) * shapingFunction->Value(srcTime,dstTime);
               if (!dstAlphaVector.IsKey(latentVariable)) dstAlphaVector.AddDat(latentVariable,hazard);
               else dstAlphaVector.GetDat(latentVariable) += hazard;
            }
         }
      }

      THash<TIntPr,TFlt> receiverPropertyGrad(propertySize());
      THash<TIntPr,TFlt> spreaderPropertyGrad(cascadeSize * propertySize());
      THash<TIntPr,TFlt> topicReceiveGrad(latentVariableSize());
      //THash<TIntPr,TFlt> acquaintanceGrad;

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (GetAcquaitance(srcNId,dstNId)!=1.0) continue;
         if (!shapingFunction->Before(srcTime,dstTime)) break; 
            
         //TFlt acquaintedValue = GetAcquaitance(srcNId, dstNId);             
         
         /*acquaintanceIndex.Val1 = srcNId; 
         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;           
            //TFlt propertyValue = MaxAlpha * sigmoid(propertyValueVector.GetDat(propertyValueIndex)); 
            TFlt propertyValue = sigmoid(propertyValueVector.GetDat(propertyValueIndex)); 
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
         }*/

         TInt maxSpreaderIndex = -1, maxReceiverIndex = -1;
         TFlt maxSpreaderValue = -DBL_MAX, maxReceiverValue = -DBL_MAX;
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
            spreaderIndex.Val2 = receiverIndex.Val2 = propertyIndex;
            TFlt spreaderValue = parameter.propertyInitValue;
            TFlt receiverValue = parameter.propertyInitValue;
            if (parameter.spreaderProperty.IsKey(spreaderIndex)) spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);
            if (parameter.receiverProperty.IsKey(receiverIndex)) receiverValue = TFlt::Abs(parameter.receiverProperty.GetDat(receiverIndex));
            if (spreaderValue > maxSpreaderValue) {
               maxSpreaderValue = spreaderValue;
               maxSpreaderIndex = propertyIndex;
            }
            if (receiverValue > maxReceiverValue) {
               maxReceiverValue = receiverValue;
               maxReceiverIndex = propertyIndex;
            }
         }
                        
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
            spreaderIndex.Val2 = receiverIndex.Val2 = propertyIndex;
            TFlt spreaderValue = parameter.propertyInitValue;
            TFlt receiverValue = parameter.propertyInitValue;
            if (parameter.spreaderProperty.IsKey(spreaderIndex)) spreaderValue = parameter.spreaderProperty.GetDat(spreaderIndex);
            if (parameter.receiverProperty.IsKey(receiverIndex)) receiverValue = parameter.receiverProperty.GetDat(receiverIndex);

            for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
               TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
               TFlt propertyValue = sigmoid(propertyValueVector.GetDat(propertyValueIndex));
               //propertyValue = propertyValue * (1.0 - propertyValue);
              
               //TRnd rnd; rnd.PutSeed(time(NULL)); 
               TFlt spreaderGrad, receiverGrad;
               if (inCascade) {
                  TFlt totalAlpha = dstAlphaVector.GetDat(latentVariable);
                  spreaderGrad =  receiverGrad = (shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/totalAlpha);
               if (propertyIndex==maxSpreaderIndex) {
                  receiverGrad *= propertyValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable)/ (TFlt)propertySize;
                  if (!receiverPropertyGrad.IsKey(receiverIndex)) receiverPropertyGrad.AddDat(receiverIndex, receiverGrad * spreaderValue);
                  else receiverPropertyGrad.GetDat(receiverIndex) += receiverGrad * spreaderValue;
               }
               }
               else {
                  receiverGrad = spreaderGrad =  shapingFunction->Integral(srcTime,dstTime);
               }

               //if (propertyIndex==maxReceiverIndex) {
                  spreaderGrad *= propertyValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable)/ (TFlt)propertySize;
                  if (!spreaderPropertyGrad.IsKey(spreaderIndex)) spreaderPropertyGrad.AddDat(spreaderIndex, spreaderGrad * receiverValue);
                  else spreaderPropertyGrad.GetDat(spreaderIndex) += spreaderGrad * receiverValue;
               //}


            }
            //printf("index:%d, %d,%d: index:%d, sValue:%f, rValue:%f, shapingVal:%f\n",datum.index(),srcNId(),dstNId(),propertyIndex(),sValue(),rValue(),shapingFunction->Integral(srcTime,dstTime)()); 
            //printf("index:%d, %d,%d: index:%d, sValue:%f, rValue:%f, shapingVal:%f\n",datum.index(),srcNId(),dstNId(),propertyIndex(),sValue(),rValue(),shapingFunction->Integral(srcTime,dstTime)()); 
         }

         TInt maxIndex = -1, minIndex = -1;
         TFlt maxValue = -DBL_MAX, minValue = DBL_MAX;
         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            TFlt Value = latentDistributions.GetDat(datum.index).GetDat(latentVariable);
            if (Value > maxValue) {
               maxValue = Value;
               maxIndex = latentVariable;
            }
            if (Value < minValue) {
               minValue = Value;
               minIndex = latentVariable;
            }
         }
         
         spreaderIndex.Val1 = srcNId; receiverIndex.Val1 = dstNId;
         for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
            spreaderIndex.Val2 = receiverIndex.Val2 = latentVariable;
            TFlt spreaderValue = parameter.topicInitValue;
            TFlt receiverValue = parameter.topicInitValue;
            if (parameter.topicSpread.IsKey(spreaderIndex)) spreaderValue = parameter.topicSpread.GetDat(spreaderIndex);
            if (parameter.topicReceive.IsKey(receiverIndex)) receiverValue = parameter.topicReceive.GetDat(receiverIndex);
               
            TIntPr propertyValueIndex; propertyValueIndex.Val1 = srcNId; propertyValueIndex.Val2 = latentVariable;
            TFlt propertyValue = sigmoid(propertyValueVector.GetDat(propertyValueIndex));
            //propertyValue = propertyValue * (1.0 - propertyValue);
               
            TFlt grad;
            if (inCascade) {
               TFlt totalAlpha = dstAlphaVector.GetDat(latentVariable);
               grad = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/totalAlpha;

            if (latentVariable==maxIndex) {
               grad *= propertyValue * spreaderValue;
               //grad *= propertyValue * spreaderValue * latentDistributions.GetDat(datum.index).GetDat(latentVariable);
               if (!topicReceiveGrad.IsKey(receiverIndex)) topicReceiveGrad.AddDat(receiverIndex,grad);
               else topicReceiveGrad.GetDat(receiverIndex) += grad;
            }
            }
            else grad = shapingFunction->Integral(srcTime,dstTime);

 
         }
      }

      //critical      
      #pragma omp critical
      {
         /*for (THash<TIntPr,TFlt>::TIter I = acquaintanceGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.acquaintance.IsKey(I.GetKey())) parameterGrad.acquaintance.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.acquaintance.GetDat(I.GetKey()) += I.GetDat();
            //printf("%d,%d acquaintance grad:%f\n",I.GetKey().Val1(),I.GetKey().Val2(),I.GetDat()());
         }*/
         for (THash<TIntPr,TFlt>::TIter I = receiverPropertyGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.receiverProperty.IsKey(I.GetKey())) parameterGrad.receiverProperty.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.receiverProperty.GetDat(I.GetKey()) += I.GetDat();
            //printf("%d,%d receiver property grad:%f\n",I.GetKey().Val1(),I.GetKey().Val2(),I.GetDat()());
         } 
         for (THash<TIntPr,TFlt>::TIter I = spreaderPropertyGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.spreaderProperty.IsKey(I.GetKey())) parameterGrad.spreaderProperty.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.spreaderProperty.GetDat(I.GetKey()) += I.GetDat();
            //printf("%d,%d spreader property grad:%f\n",I.GetKey().Val1(),I.GetKey().Val2(),I.GetDat()());
         } 
         for (THash<TIntPr,TFlt>::TIter I = topicReceiveGrad.BegI(); !I.IsEnd(); I++) {
            if (!parameterGrad.topicReceive.IsKey(I.GetKey())) parameterGrad.topicReceive.AddDat(I.GetKey(),I.GetDat());
            else parameterGrad.topicReceive.GetDat(I.GetKey()) += I.GetDat();
            //printf("%d,%d topic receive grad:%f\n",I.GetKey().Val1(),I.GetKey().Val2(),I.GetDat()());
         } 
      }
   }

   for (TInt latentVariable=0; latentVariable<latentVariableSize; latentVariable++) {
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

void UserPropertyFunction::updateAcquaintance() {
   THash<TIntPr,TFlt>& acquaintance = parameter.acquaintance;
   THash<TInt,TFlt> inThresholdHash, outThresholdHash;
   THash<TInt,TFlt> inEdgeNumHash, outEdgeNumHash;
   THash<TIntPr,TInt> countHash;

   TInt support = 0;
   TFlt totalPropertyValue = 0.0;
   for (THash<TIntPr,TFlt>::TIter EI = acquaintance.BegI(); !EI.IsEnd(); EI++) {
      TInt srcNId = EI.GetKey().Val1, dstNId = EI.GetKey().Val2;
      countHash.AddDat(EI.GetKey(), allPossibleEdges.GetDat(dstNId).GetDat(srcNId));

      TFlt propertyValue = GetPropertyValue(srcNId, dstNId);
      totalPropertyValue = totalPropertyValue + propertyValue;
      if (!inThresholdHash.IsKey(dstNId)) inThresholdHash.AddDat(dstNId, propertyValue);
      else inThresholdHash.GetDat(dstNId) += propertyValue;
      if (!inEdgeNumHash.IsKey(dstNId)) inEdgeNumHash.AddDat(dstNId, 1.0);
      else inEdgeNumHash.GetDat(dstNId)++;

      if (!outThresholdHash.IsKey(srcNId)) outThresholdHash.AddDat(srcNId, propertyValue);
      else outThresholdHash.GetDat(srcNId) += propertyValue;
      if (!outEdgeNumHash.IsKey(srcNId)) outEdgeNumHash.AddDat(srcNId, 1.0);
      else outEdgeNumHash.GetDat(srcNId)++;
   }

   for (THash<TInt,TFlt>::TIter TI = inThresholdHash.BegI(); !TI.IsEnd(); TI++) {
      TI.GetDat() = TI.GetDat() / inEdgeNumHash.GetDat(TI.GetKey()) + TMath::Log(inEdgeNumHash.GetDat(TI.GetKey()));
   } 

   countHash.SortByDat(false);
   TInt size = acquaintance.Len();
   TInt supportRange = size / 2;
   for (TInt i=0; i<supportRange; i++) {
      support += countHash[i];
   }

   TFlt averagePropertyValue = totalPropertyValue / TFlt(size);
   support = support / supportRange; 
   TInt addedEdgeNum = 0, removedEdgeNum = 0;

   for (THash<TInt, THash<TInt, TInt> >::TIter MI = allPossibleEdges.BegI(); !MI.IsEnd(); MI++) {
      THash<TInt, TInt>& map = MI.GetDat();
      TInt dstNId = MI.GetKey(), srcNId;
      TFlt maxPropertyValue = -DBL_MAX;
      TInt maxSrcNId = -1;
      for (THash<TInt,TInt>::TIter EI = map.BegI(); !EI.IsEnd(); EI++) {
         srcNId = EI.GetKey();
         TIntPr index; index.Val1 = srcNId; index.Val2 = dstNId;
         //if (acquaintance.IsKey(index)) continue;
         if (acquaintance.IsKey(index) || EI.GetDat() <= support) continue;

         TFlt propertyValue = GetPropertyValue(srcNId, dstNId), threshold = averagePropertyValue;
         if (inThresholdHash.IsKey(dstNId)) threshold = inThresholdHash.GetDat(dstNId);
         /*if (outThresholdHash.IsKey(srcNId)) { 
            TFlt outThreshold = outThresholdHash.GetDat(srcNId) / outEdgeNumHash.GetDat(srcNId) + TMath::Log(outEdgeNumHash.GetDat(srcNId));
            if (threshold < outThreshold) threshold = outThreshold;
         }*/

         if (propertyValue > maxPropertyValue && propertyValue > threshold) {
            maxPropertyValue = propertyValue;
            maxSrcNId = srcNId;
         }
      }

      if (maxSrcNId!=-1) {
         TIntPr index; index.Val1 = maxSrcNId; index.Val2 = dstNId;
         TFlt propertyValue = GetPropertyValue(maxSrcNId, dstNId);
         if (!outEdgeNumHash.IsKey(maxSrcNId)) {
            outEdgeNumHash.AddDat(maxSrcNId, 1.0);
            outThresholdHash.AddDat(maxSrcNId, 0.0);
         }
         outEdgeNumHash.GetDat(maxSrcNId)++;
         outThresholdHash.GetDat(maxSrcNId) += propertyValue;
         acquaintance.AddDat(index, 1.0);
         addedEdgeNum++;
         printf("add edge %d,%d \n", maxSrcNId(), dstNId());
      }
      
   }
   
   /*possibleHash.SortByDat();
   size = possibleHash.Len();
   range = size / 20;
   for (TInt i=0; i<range; i++) {
      downTotalPropertyValue += possibleHash[i];
   }
   TFlt downAveragePropertyValue = downTotalPropertyValue / TFlt(range);
   size = acquaintanceHash.Len();

   for (TInt i=size-1;i>=0;i--) {
      if (acquaintanceHash[i] < downAveragePropertyValue) {
         TIntPr index = acquaintanceHash.GetKey(i);
         acquaintance.DelKey(index);
         removedEdgeNum++;
         printf("remove edge %d,%d \n", index.Val1(), index.Val2());
      }
   }*/

   printf("average property value %f, support %d\n", averagePropertyValue(), support());
   printf("new added edges number %d, new removed edges number %d\n", addedEdgeNum(), removedEdgeNum());
}

void UserPropertyFunction::initParameter(Data data, UserPropertyFunctionConfigure configure) {
   parameter.init(data,configure);
   allPossibleEdgeNum = 0;
   for (THash<TInt, TCascade>::TIter CI = data.cascH.BegI(); !CI.IsEnd(); CI++) {
      const TCascade& cascade = CI.GetDat();
      TIntPr index;
      for (THash<TInt, THitInfo>::TIter DNI = cascade.BegI(); DNI < cascade.EndI(); DNI++) {
         index.Val2 = DNI.GetKey();
         if (!allPossibleEdges.IsKey(index.Val2)) allPossibleEdges.AddDat(index.Val2, THash<TInt,TInt>());
         THash<TInt,TInt>& possibleEdges = allPossibleEdges.GetDat(index.Val2);

         for (THash<TInt, THitInfo>::TIter SNI = cascade.BegI(); SNI < DNI; SNI++) {
            index.Val1 = SNI.GetKey();
            if (!possibleEdges.IsKey(index.Val1)) {
               possibleEdges.AddDat(index.Val1, 1);
               allPossibleEdgeNum++;
            }
            else possibleEdges.GetDat(index.Val1)++;
         }
      } 
      THash<TInt, THitInfo>::TIter SNI = cascade.BegI();
      index.Val1 = SNI.GetKey();
      SNI++;
      index.Val2 = SNI.GetKey();
      if (!parameter.acquaintance.IsKey(index)) parameter.acquaintance.AddDat(index, 1.0);
   }

   printf("all possible edges: %d, all initial edges: %d\n", allPossibleEdgeNum(), parameter.acquaintance.Len()); 
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
   updateRProp(initVal, lr.receiverProperty, gradient.receiverProperty);
   updateRProp(initVal, lr.spreaderProperty, gradient.spreaderProperty);
   updateRProp(initVal, lr.topicReceive, gradient.topicReceive);
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
   updateRMSProp(alpha, lr.receiverProperty, gradient.receiverProperty);
   updateRMSProp(alpha, lr.spreaderProperty, gradient.spreaderProperty);
   updateRMSProp(alpha, lr.topicReceive, gradient.topicReceive);
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
   updateAverageRMSProp(alpha, sigmaes[1], gradient.receiverProperty);
   updateAverageRMSProp(alpha, sigmaes[2], gradient.spreaderProperty);
   updateAverageRMSProp(alpha, sigmaes[3], gradient.topicReceive);
}

UserPropertyParameter::UserPropertyParameter() {
   reset();
}

UserPropertyParameter& UserPropertyParameter::operator = (const UserPropertyParameter& p) {
   reset();
   propertyInitValue = p.propertyInitValue;
   propertyMaxValue = p.propertyMaxValue;
   propertyMinValue = p.propertyMinValue;
   topicInitValue = p.topicInitValue;
   topicMaxValue = p.topicMaxValue;
   topicMinValue = p.topicMinValue;
   //acquaintanceInitValue = p.acquaintanceInitValue;
   //acquaintanceMaxValue = p.acquaintanceMaxValue;
   //acquaintanceMinValue = p.acquaintanceMinValue;
   //acquaintance = p.acquaintance;
   receiverProperty = p.receiverProperty;
   spreaderProperty = p.spreaderProperty;
   topicReceive = p.topicReceive;
   topicSpread = p.topicSpread;
   kPi = p.kPi;
   kPi_times = p.kPi_times;
   return *this; 
}

void UserPropertyParameter::AddEqualTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src) {
   for (THash<TIntPr,TFlt>::TIter PI = src.BegI(); !PI.IsEnd(); PI++) {
      TIntPr key = PI.GetKey();
      TFlt value = PI.GetDat();
      if (!dst.IsKey(key)) dst.AddDat(key,value);
      else dst.GetDat(key) += value;
      //printf("%d,%d: += value:%f, m:%f\n", PI.GetDat(), key.Val1(), key.Val2(), PI.GetDat(), m);
   }
}

UserPropertyParameter& UserPropertyParameter::operator += (const UserPropertyParameter& p) {
   //AddEqualTHash(acquaintance, p.acquaintance);
   AddEqualTHash(receiverProperty, p.receiverProperty);
   AddEqualTHash(spreaderProperty, p.spreaderProperty);
   AddEqualTHash(topicReceive, p.topicReceive);
   AddEqualTHash(topicSpread, p.topicSpread);
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

void UserPropertyParameter::MultiplyTHash(THash<TIntPr,TFlt>& dst, const TFlt multiplier) {
   for (THash<TIntPr, TFlt>::TIter VI = dst.BegI(); !VI.IsEnd(); VI++) {
      VI.GetDat() *= multiplier;
   }
}

UserPropertyParameter& UserPropertyParameter::operator *= (const TFlt multiplier) {
   //MultiplyTHash(acquaintance, multiplier);
   MultiplyTHash(receiverProperty, multiplier);
   MultiplyTHash(spreaderProperty, multiplier);
   MultiplyTHash(topicReceive, multiplier);
   MultiplyTHash(topicSpread, multiplier);
   return *this; 
}

void UserPropertyParameter::UpdateTHash(THash<TIntPr,TFlt>& dst, const THash<TIntPr,TFlt>& src, TFlt minValue, TFlt initValue, TFlt maxValue, TStr comment) {
   for (THash<TIntPr,TFlt>::TIter PI = src.BegI(); !PI.IsEnd(); PI++) {
      TIntPr key = PI.GetKey();
      TFlt valueGradient = PI.GetDat(), value;
      //printf("%s %d,%d: %f, dat:%f, m:%f\n", comment(), key.Val1(), key.Val2(), value(), PI.GetDat()(),m());
      if (dst.IsKey(key)) value = dst.GetDat(key);
      else value = initValue;

      if (comment != TStr("acquaintance"))
         value -= (valueGradient + (Regularizer ? Mu : TFlt(0.0)) * value);
      else
         value -= valueGradient;

      if (value < minValue) value = minValue;
      if (value > maxValue) value = maxValue;

      if (!dst.IsKey(key)) dst.AddDat(key,value);
      else dst.GetDat(key) = value;
      //printf("%s %d,%d: %f, updated\n", comment(), key.Val1(), key.Val2(), dst.GetDat(key)());
   }
}

UserPropertyParameter& UserPropertyParameter::projectedlyUpdateGradient(const UserPropertyParameter& p) {
   //UpdateTHash(acquaintance, p.acquaintance, acquaintanceMinValue, acquaintanceInitValue, acquaintanceMaxValue, "acquaintance");
   UpdateTHash(receiverProperty, p.receiverProperty, propertyMinValue, propertyInitValue, propertyMaxValue, "receiver property");
   UpdateTHash(spreaderProperty, p.spreaderProperty, 0.001, propertyInitValue, propertyMaxValue, "spreader property");
   UpdateTHash(topicReceive, p.topicReceive, topicMinValue, topicInitValue, topicMaxValue, "topic receiver");
   //UpdateTHash(topicSpread, p.topicSpread, topicMinValue, topicInitValue, topicMaxValue, "topic spreader");
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
   //acquaintance.Clr();
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
   topicStdValue = configure.topicStdValue;
   topicMaxValue = configure.topicMaxValue;
   topicMinValue = configure.topicMinValue;
   
   //acquaintanceInitValue = configure.acquaintanceInitValue;
   //acquaintanceMaxValue = configure.acquaintanceMaxValue;
   //acquaintanceMinValue = configure.acquaintanceMinValue;
  
   MaxAlpha = configure.MaxAlpha;
   MinAlpha = configure.MinAlpha;
   propertySize = configure.propertySize;
   
   Regularizer = configure.Regularizer;
   Mu = configure.Mu; 
 
   TRnd rnd; rnd.PutSeed(time(NULL));
   for (TInt i=0;i<configure.topicSize;i++) {
      kPi.AddDat(i,rnd.GetUniDevInt(1,2));
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
        //spreaderProperty.AddDat(i,rnd.GetNrmDev(configure.propertyInitValue(), 0.1, configure.propertyMinValue(), configure.propertyMaxValue()));
        spreaderProperty.AddDat(i,rnd.GetRayleigh(0.01));
        receiverProperty.AddDat(i,rnd.GetNrmDev(configure.propertyInitValue(), 0.01, configure.propertyMinValue(), configure.propertyMaxValue()));
        //printf("%d,%d receiverProperty:%f, spreaderProperty:%f\n",i.Val1(),i.Val2(),receiverProperty.GetDat(i)(),spreaderProperty.GetDat(i)());
     }

     for (TInt topic=0; topic<configure.topicSize; topic++) {
        TIntPr i; i.Val1 = NI.GetKey(); i.Val2 = topic;
        topicSpread.AddDat(i,1.0);
        topicReceive.AddDat(i,rnd.GetNrmDev(configure.topicInitValue(), configure.topicStdValue(), configure.topicMinValue(), configure.topicMaxValue()));
        //printf("%d,%d topicReceive:%f, topicSpread:%f\n",i.Val1(),i.Val2(),topicReceive.GetDat(i)(),topicSpread.GetDat(i)());
     }
  }

}

void UserPropertyParameter::GenParameters(TStrFltFltHNEDNet& network, UserPropertyFunctionConfigure configure, TInt edgeNums) {
   set(configure);
   TRnd rnd; rnd.PutSeed(time(NULL));
   for (TStrFltFltHNEDNet::TNodeI NI = network.BegNI(); NI < network.EndNI(); NI++) {
      TIntPr i; i.Val1 = NI.GetId();

      for (TInt index=0; index<configure.propertySize; index++) {
         i.Val2 = index;
         //spreaderProperty.AddDat(i,rnd.GetNrmDev(configure.propertyInitValue(), 2.0, configure.propertyMinValue(), configure.propertyMaxValue()));
         spreaderProperty.AddDat(i,rnd.GetRayleigh(0.5));
         receiverProperty.AddDat(i,rnd.GetNrmDev(configure.propertyInitValue(), 2.0, configure.propertyMinValue(), configure.propertyMaxValue()));
      }
     
      for (TInt topic=0; topic<configure.topicSize; topic++) {
         i.Val2 = topic;
         topicSpread.AddDat(i,1.0);
         topicReceive.AddDat(i,rnd.GetNrmDev(configure.topicInitValue(), configure.topicStdValue(), configure.topicMinValue(), configure.topicMaxValue()));
      }
   }

   THash<TIntPr,TFlt> propertyHash;
  
   for (TStrFltFltHNEDNet::TNodeI SNI = network.BegNI(); SNI < network.EndNI(); SNI++) {
      for (TStrFltFltHNEDNet::TNodeI DNI = network.BegNI(); DNI < network.EndNI(); DNI++) {
         if (SNI==DNI) continue;
         TIntPr index; index.Val1 = SNI.GetId(); index.Val2 = DNI.GetId();
         TFlt propertyValue = GetPropertyValue(index.Val1, index.Val2);
         propertyHash.AddDat(index, propertyValue);
      }
   }

   propertyHash.SortByDat(false);
 
   for (TInt i=0; i<edgeNums; i++) {
      TIntPr index = propertyHash.GetKey(i);
      network.AddEdge(index.Val1, index.Val2, TFltFltH());
   } 

   for (TStrFltFltHNEDNet::TEdgeI EI = network.BegEI(); EI < network.EndEI(); EI++) {
      TIntPr i; i.Val1 = EI.GetSrcNId(); i.Val2 = EI.GetDstNId();
      acquaintance.AddDat(i, 1.0);
   }
}

TFlt UserPropertyParameter::GetAcquaitance(TInt srcNId, TInt dstNId) const {
   TIntPr acquaintanceIndex;
   acquaintanceIndex.Val1 = srcNId; acquaintanceIndex.Val2 = dstNId;
   if (acquaintance.IsKey(acquaintanceIndex)) return 1.0;
   return 0.0; 
}

TFlt UserPropertyParameter::GetPropertyValue(TInt srcNId, TInt dstNId) const {
   TFlt alpha = 0.0;
   TIntPr receiverIndex, spreaderIndex;
   receiverIndex.Val1 = dstNId; spreaderIndex.Val1 = srcNId;
   
   for (TInt propertyIndex=0; propertyIndex<propertySize; propertyIndex++) {
      receiverIndex.Val2 = spreaderIndex.Val2 = propertyIndex;
      TFlt spreaderValue = propertyInitValue, receiverValue = propertyInitValue;
      if (spreaderProperty.IsKey(spreaderIndex)) spreaderValue = spreaderProperty.GetDat(spreaderIndex);
      if (receiverProperty.IsKey(receiverIndex)) receiverValue = receiverProperty.GetDat(receiverIndex);
         alpha += spreaderValue * receiverValue;
   }
   alpha /= (TFlt)propertySize;
   return alpha;
}

TFlt UserPropertyParameter::GetTopicValue(TInt srcNId, TInt dstNId, TInt topic) const {
   TIntPr receiverIndex, spreaderIndex;
   receiverIndex.Val1 = dstNId; spreaderIndex.Val1 = srcNId;
   receiverIndex.Val2 = spreaderIndex.Val2 = topic; 
   TFlt spreaderValue = topicInitValue, receiverValue = topicInitValue;
   if (topicSpread.IsKey(spreaderIndex)) spreaderValue = topicSpread.GetDat(spreaderIndex);
   if (topicReceive.IsKey(receiverIndex)) receiverValue = topicReceive.GetDat(receiverIndex);
   return spreaderValue * receiverValue;
}

TFlt UserPropertyParameter::GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const {
   if (GetAcquaitance(srcNId,dstNId)!=1.0) return 0.0;
   TFlt propertyValue = GetPropertyValue(srcNId,dstNId);
   //TFlt acquaintedValue = GetAcquaitance(srcNId,dstNId); 
   TFlt topicValue = GetTopicValue(srcNId,dstNId,topic);      
            
   TFlt alpha = ReLU(propertyValue + topicValue);
   //alpha = acquaintedValue * MaxAlpha * sigmoid(alpha);
   if (alpha > MaxAlpha) alpha = MaxAlpha;
   else if (alpha < 0.0001) alpha = 0.0001;
   return alpha;
}
