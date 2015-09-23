#include <DecayCascadesFunction.h>

TFlt DecayCascadesFunction::JointLikelihood(Datum datum, TInt latentVariable) const {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
   double totalLoss = 0.0;

   int nodeSize = NodeNmH.Len();
   #pragma omp parallel for reduction(+:totalLoss)
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      TInt dstNId = key, srcNId;
      TFlt sumInLog = 0.0, val = 0.0;
      TFlt dstTime, srcTime;
      double lossValue;

      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) dstTime = Cascade.GetTm(dstNId);
      else dstTime = Cascade.GetMaxTm() + observedWindow;

      TFlt nodePosition = 0.0;
      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++, nodePosition++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
                        
         TFlt alpha = 0.0;
         TIntPr key(srcNId, dstNId);
         if (potentialEdges.IsKey(key))
            alpha = GetTopicAlpha(srcNId, dstNId, latentVariable) / TMath::Power(decayRatio, nodePosition);

         sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
         val += alpha * shapingFunction->Integral(srcTime,dstTime);
         //printf("sumInLog:%f,val:%f, alpha:%f, shapingVal:%f, shapingInt:%f\n",sumInLog(),val(),alpha(),shapingFunction->Value(srcTime,dstTime)(),shapingFunction->Integral(srcTime,dstTime)()); 
      }
      lossValue = val;
      
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime && sumInLog!=0.0) lossValue -= TMath::Log(sumInLog);
      totalLoss += lossValue;
   }

   TFlt logPi = TMath::Log(parameter.priorTopicProbability.GetDat(latentVariable));
   //printf("datum:%d, Myloss:%f, logPi:%f\n",datum.index(), totalLoss, logPi());
   return logPi - totalLoss;
}

DecayCascadesParameter& DecayCascadesFunction::gradient(Datum datum) {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
 
   parameterGrad.reset();
   if (parameterGrad.priorTopicProbability.Empty()) {
      parameterGrad.sampledTimes = 0;;
      for (TInt i = 0; i < parameter.latentVariableSize; i++) {
         parameterGrad.priorTopicProbability.AddDat(i, 0.0);
      }
   }

   for (TInt i = 0; i < parameter.latentVariableSize; i++) {
      parameterGrad.kAlphas.AddDat(i, THash<TIntPr, TFlt>());
      parameterGrad.priorTopicProbability.GetDat(i) += latentDistributions.GetDat(datum.index).GetDat(i);
   }
   parameterGrad.sampledTimes++;

   int nodeSize = NodeNmH.Len();
   #pragma omp parallel for
   for (int i=0; i<nodeSize; i++) {
      TInt dstNId = NodeNmH.GetKey(i), srcNId;
      TFlt dstTime, srcTime;
      THash<TInt,TFlt> dstAlphas;

      for (TInt i = 0; i < parameter.latentVariableSize; i++) dstAlphas.AddDat(i, 0.0);

      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) {
         dstTime = Cascade.GetTm(dstNId);
         TFlt nodePosition = 0.0;
         for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++, nodePosition++) {
            srcNId = CascadeNI.GetKey();
            srcTime = CascadeNI.GetDat().Tm;

            if (!shapingFunction->Before(srcTime,dstTime)) break; 
                         
            for (TInt i = 0; i < parameter.latentVariableSize; i++) {
               TFlt alpha = parameter.GetTopicAlpha(srcNId, dstNId, i) / TMath::Power(decayRatio, nodePosition);
               dstAlphas.GetDat(i) += alpha * shapingFunction->Value(srcTime,dstTime);
               //printf("sumInLog:%f, alpha:%f, val:%f, initAlpha:%f\n",sumInLog(),alpha(),shapingFunction->Value(srcTime,dstTime)(),parameter.InitAlpha());
            }
         }
      }
      else dstTime = Cascade.GetMaxTm() + observedWindow;
   
      THash<TInt, THash<TIntPr, TFlt> > kAlphasGradient;
      for (TInt i = 0; i < parameter.latentVariableSize; i++) {
         kAlphasGradient.AddDat(i, THash<TIntPr, TFlt>());
      }

      TFlt nodePosition = 0.0;
      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++, nodePosition++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;
   
         if (!shapingFunction->Before(srcTime,dstTime)) break; 
         TIntPr key(srcNId, dstNId);
         if (!potentialEdges.IsKey(key)) continue;
                           
         TIntPr alphaIndex; alphaIndex.Val1 = srcNId; alphaIndex.Val2 = dstNId;
         TFlt val = 0.0;

         for (TInt i = 0; i < parameter.latentVariableSize; i++) {
            if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime)
               val = (shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime) / dstAlphas.GetDat(i)) / TMath::Power(decayRatio, nodePosition);
            else
               val = shapingFunction->Integral(srcTime,dstTime) / TMath::Power(decayRatio, nodePosition);
            THash<TIntPr, TFlt>& alphaGradient = kAlphasGradient.GetDat(i);
            alphaGradient.AddDat(alphaIndex, val * latentDistributions.GetDat(datum.index).GetDat(i));
         }
         //printf("index:%d, %d,%d: gradient:%f, shapingVal:%f, sumInLog:%f\n",datum.index(),srcNId(),dstNId(),val(),shapingFunction->Integral(srcTime,dstTime)(),sumInLog()); 
      }

      #pragma omp critical 
      {
         for (TInt i = 0; i < parameter.latentVariableSize; i++) {
            THash<TIntPr, TFlt>& alphaGradient = kAlphasGradient.GetDat(i);
            THash<TIntPr, TFlt>& alpha = parameterGrad.kAlphas.GetDat(i);
            for (THash<TIntPr, TFlt>::TIter AI = alphaGradient.BegI(); !AI.IsEnd(); AI++) {
               if (alpha.IsKey(AI.GetKey())) alpha.GetDat(AI.GetKey()) += AI.GetDat();
               else alpha.AddDat(AI.GetKey(), AI.GetDat());
            }
         }
      }
   }

   return parameterGrad;
}

void DecayCascadesFunction::maximize() {
   if (parameterGrad.sampledTimes == 0.0) return;
   for (THash<TInt,TFlt>::TIter VI = parameterGrad.priorTopicProbability.BegI(); !VI.IsEnd(); VI++) {
      //printf("topic %d, value %f, ", VI.GetKey()(), VI.GetDat()());
      parameter.priorTopicProbability.GetDat(VI.GetKey()) = VI.GetDat() / parameterGrad.sampledTimes;
      if (parameter.priorTopicProbability.GetDat(VI.GetKey()) < 0.001) parameter.priorTopicProbability.GetDat(VI.GetKey()) = 0.001; 
      VI.GetDat() = 0.0;
   }
   parameterGrad.sampledTimes = 0.0;
}

void DecayCascadesFunction::set(DecayCascadesFunctionConfigure configure) {
   latentVariableSize = configure.latentVariableSize;
   shapingFunction = configure.shapingFunction;
   decayRatio = configure.decayRatio;
   parameter.set(configure);
   parameterGrad.set(configure);
}

void DecayCascadesFunction::init(Data data, TInt NodeNm) {
   parameter.init(data, NodeNm);
}

void DecayCascadesParameter::set(DecayCascadesFunctionConfigure configure) {
   Regularizer = configure.Regularizer;
   Mu = configure.Mu;
   Tol = configure.Tol;
   InitAlpha = configure.InitAlpha;
   MaxAlpha = configure.MaxAlpha;
   MinAlpha = configure.MinAlpha;
   latentVariableSize = configure.latentVariableSize;
}

void DecayCascadesParameter::init(Data data, TInt NodeNm) {
   for (TInt i=0; i < latentVariableSize; i++) {
      kAlphas.AddDat(i, THash<TIntPr, TFlt>());
   }
}

void DecayCascadesParameter::initPriorTopicProbabilityParameter() {
   TFlt::Rnd.PutSeed(0);
   TFlt sum = 0.0;
   for (TInt i=0; i < latentVariableSize; i++) {
      priorTopicProbability.AddDat(i, TFlt::Rnd.GetUniDev());
      sum += priorTopicProbability.GetDat(i);
   }
   for (TInt i=0; i < latentVariableSize; i++) priorTopicProbability.GetDat(i) = priorTopicProbability.GetDat(i) / sum;
}

void DecayCascadesParameter::initAlphaParameter() {
   for (TInt i=0; i < latentVariableSize; i++) {
      THash<TIntPr, TFlt>& alphas = kAlphas.GetDat(i);
      for (THash<TIntPr, TFlt>::TIter AI = alphas.BegI(); !AI.IsEnd(); AI++) {
         AI.GetDat() = TFlt::Rnd.GetUniDev() * (MaxAlpha - MinAlpha) + MinAlpha;
      }
   }
}

void DecayCascadesFunction::initPotentialEdges(Data data) {
  THash<TInt, TCascade>& cascades = data.cascH;
  int cascadesNum = cascades.Len();
  //#pragma omp parallel for
  for (int i=0;i<cascadesNum;i++) {
     TCascade& cascade = cascades[i];
     for (THash<TInt, THitInfo>::TIter srcNI = cascade.BegI(); srcNI < cascade.EndI(); srcNI++) {
        for (THash<TInt, THitInfo>::TIter dstNI = srcNI; dstNI < cascade.EndI(); dstNI++) {
           if (srcNI==dstNI) continue;
           TIntPr key(srcNI.GetKey(), dstNI.GetKey());
           if (dstNI.GetDat().Tm <= data.time)
              potentialEdges.AddDat(key, 1.0);
        } 
     }
  }
}

void DecayCascadesParameter::reset() {
   kAlphas.Clr();
}

DecayCascadesParameter& DecayCascadesParameter::operator = (const DecayCascadesParameter& p) {

   kAlphas.Clr();
   kAlphas = p.kAlphas;

   priorTopicProbability.Clr();
   priorTopicProbability = p.priorTopicProbability;

   sampledTimes = p.sampledTimes; 
   return *this;
}

DecayCascadesParameter& DecayCascadesParameter::operator += (const DecayCascadesParameter& p) {
   for(THash<TInt, THash<TIntPr, TFlt> >::TIter AI = p.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      if (!kAlphas.IsKey(key)) {
         kAlphas.AddDat(key, THash<TIntPr, TFlt>());
      }

      THash<TIntPr, TFlt>& alphas = kAlphas.GetDat(key);
      for (THash<TIntPr,TFlt>::TIter aI = AI.GetDat().BegI(); !aI.IsEnd(); aI++) {
         TIntPr alphaIndex = aI.GetKey();
         TFlt alpha = aI.GetDat();
         if (!alphas.IsKey(alphaIndex)) alphas.AddDat(alphaIndex, alpha);
         else alphas.GetDat(alphaIndex) += alpha;
         //printf("topic: %d, %d,%d: += alpha:%f\n", key(), alphaIndex.Val1(), alphaIndex.Val2(), aI.GetDat());
      }
   }
   return *this;
}

DecayCascadesParameter& DecayCascadesParameter::operator *= (const TFlt multiplier) {
   for(THash<TInt, THash<TIntPr,TFlt> >::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      THash<TIntPr, TFlt>& alphas = AI.GetDat();
      for (THash<TIntPr,TFlt>::TIter aI = alphas.BegI(); !aI.IsEnd(); aI++) aI.GetDat() *= multiplier;
   }
   return *this;
}

DecayCascadesParameter& DecayCascadesParameter::projectedlyUpdateGradient(const DecayCascadesParameter& p) {
   for(THash<TInt, THash<TIntPr,TFlt> >::TIter AI = p.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      THash<TIntPr,TFlt>& alphas = kAlphas.GetDat(key);
      for (THash<TIntPr,TFlt>::TIter aI = AI.GetDat().BegI(); !aI.IsEnd(); aI++) {
         TIntPr alphaIndex = aI.GetKey();
         TFlt alphaGradient = aI.GetDat(), alpha, value;
         if (alphas.IsKey(alphaIndex)) value = alphas.GetDat(alphaIndex); 
         else value = InitAlpha;

         alpha = value - (alphaGradient + (Regularizer ? Mu : TFlt(0.0)) * alpha);

         if (alpha < Tol) alpha = Tol;
         if (alpha > MaxAlpha) alpha = MaxAlpha;

         if (!alphas.IsKey(alphaIndex)) alphas.AddDat(alphaIndex, alpha);
         else alphas.GetDat(alphaIndex) = alpha;
         //printf("topic: %d, %d,%d: alpha %f -> %f , gradient %f\n", key(), alphaIndex.Val1(), alphaIndex.Val2(), value(), alpha(), alphaGradient());
      }
   }
   return *this;
}

TFlt DecayCascadesParameter::GetTopicAlpha(TInt srcNId, TInt dstNId, TInt topic) const {
   const THash<TIntPr, TFlt>& alphas = kAlphas.GetDat(topic);
   TIntPr index(srcNId,dstNId);
   if (alphas.IsKey(index)) return alphas.GetDat(index);
   return InitAlpha;
}

TFlt DecayCascadesParameter::GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const {
  const THash<TIntPr,TFlt>& alphas = kAlphas.GetDat(topic);
  TFlt alpha = 0.0;
  TIntPr index(srcNId,dstNId);
  if (!alphas.IsKey(index)) return alpha;
  return alphas.GetDat(index); 
}
