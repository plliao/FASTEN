#include <MMRateFunction.h>

TFlt MMRateFunction::JointLikelihood(Datum datum, TInt latentVariable) const {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
   TFlt totalLoss = 0.0;
   TFlt diffusionPattern;
   if (parameter.diffusionPatterns.IsKey(datum.index)) diffusionPattern = parameter.diffusionPatterns.GetDat(datum.index);
   else diffusionPattern = parameter.InitDiffusionPattern;
   const THash<TIntPr, TFlt>& alphas = parameter.kAlphas.GetDat(latentVariable);

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
                        
         TIntPr alphaIndex; alphaIndex.Val1 = srcNId; alphaIndex.Val2 = dstNId;

         TFlt alpha;
         if (alphas.IsKey(alphaIndex)) alpha = alphas.GetDat(alphaIndex);
         else alpha = parameter.InitAlpha;
         alpha += diffusionPattern;

         sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
         val += alpha * shapingFunction->Integral(srcTime,dstTime);
         //printf("sumInLog:%f,val:%f, alpha:%f, shapingVal:%f, shapingInt:%f\n",sumInLog(),val(),alpha(),shapingFunction->Value(srcTime,dstTime)(),shapingFunction->Integral(srcTime,dstTime)()); 
      }
      lossTable[i] = val;
      
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime && sumInLog!=0.0) lossTable[i] -= TMath::Log(sumInLog);
      
   }

   for (int i=0;i<nodeSize;i++) totalLoss += lossTable[i];
   delete[] lossTable;

   //printf("datum:%d, Myloss:%f\n",datum.index(), totalLoss());
   TFlt logP = -1.0 * totalLoss;
   TFlt logPi = TMath::Log(parameter.kPi.GetDat(latentVariable));
   //printf("logP: %f, logPi=%f\n",logP(),logPi());
   return logP + logPi;
}

MMRateParameter& MMRateFunction::gradient(Datum datum) {
   parameterGrad.reset();
      
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
   TFlt diffusionPattern;
   if (parameter.diffusionPatterns.IsKey(datum.index)) diffusionPattern = parameter.diffusionPatterns.GetDat(datum.index);
   else diffusionPattern = parameter.InitDiffusionPattern;

   for (THash<TInt, THash<TIntPr, TFlt> >::TIter AI = parameter.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      THash<TIntPr, TFlt>& alphasGradient = parameterGrad.kAlphas.GetDat(key);
      THash<TIntPr, TFlt>& alphas = parameter.kAlphas.GetDat(key);

      int nodeSize = NodeNmH.Len();
      int cascadeSize = Cascade.Len();
      int *srcNIds = new int[nodeSize * cascadeSize];
      int *dstNIds = new int[nodeSize * cascadeSize];
      float *vals  = new float[nodeSize * cascadeSize];
      float *diffusionPatternVals = new float[nodeSize];
    
      for (int i=0;i<nodeSize;i++) {
         diffusionPatternVals[i] = 0.0;
         for (int j=0;j<cascadeSize;j++) {
            int index = i*cascadeSize + j;
            srcNIds[index] = dstNIds[index] = -1;
            vals[index] = -1.0;
         }
      }
   
      #pragma omp parallel for
      for (int i=0;i<nodeSize;i++) {
         TInt dstNId = NodeNmH.GetKey(i), srcNId;
         TFlt sumInLog = 0.0, val = 0.0;
         TFlt dstTime, srcTime;
   
         if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) {
            dstTime = Cascade.GetTm(dstNId);
            for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
               srcNId = CascadeNI.GetKey();
               srcTime = CascadeNI.GetDat().Tm;
   
               if (!shapingFunction->Before(srcTime,dstTime)) break; 
                           
               TIntPr alphaIndex; alphaIndex.Val1 = srcNId; alphaIndex.Val2 = dstNId;
   
               TFlt alpha;
               if (alphas.IsKey(alphaIndex)) alpha = alphas.GetDat(alphaIndex);
               else alpha = parameter.InitAlpha;
               alpha += diffusionPattern;
            
               sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
               //printf("sumInLog:%f, alpha:%f, val:%f, initAlpha:%f\n",sumInLog(),alpha(),shapingFunction->Value(srcTime,dstTime)(),parameter.InitAlpha());
            }
         }
         else dstTime = CurrentTime;
   
         int j=0;
         for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++,j++) {
            srcNId = CascadeNI.GetKey();
            srcTime = CascadeNI.GetDat().Tm;
   
            if (!shapingFunction->Before(srcTime,dstTime)) break; 
                           
            TIntPr alphaIndex; alphaIndex.Val1 = srcNId; alphaIndex.Val2 = dstNId;
            if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime)
               val = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/sumInLog;
            else
               val = shapingFunction->Integral(srcTime,dstTime);
               
            int index = i*cascadeSize + j;
            srcNIds[index] = srcNId();
            dstNIds[index] = dstNId();
            vals[index] = val();
            diffusionPatternVals[i] += val();
            //printf("index:%d, %d,%d: gradient:%f, shapingVal:%f, sumInLog:%f\n",datum.index(),srcNId(),dstNId(),val(),shapingFunction->Integral(srcTime,dstTime)(),sumInLog()); 
         }
      }
   
      float diffusionPatternGradient = 0.0;
      for (int i=0;i<nodeSize;i++) {
         diffusionPatternGradient += diffusionPatternVals[i];
         for (int j=0;j<cascadeSize;j++) {
            int index = i*cascadeSize + j;
            if (srcNIds[index]==-1) break;
            TIntPr alphaIndex; alphaIndex.Val1 = srcNIds[index]; alphaIndex.Val2 = dstNIds[index];
            alphasGradient.AddDat(alphaIndex, vals[index] * latentDistributions.GetDat(datum.index).GetDat(key));
         }
      }
      diffusionPatternGradient *= latentDistributions.GetDat(datum.index).GetDat(key);
      if (!parameterGrad.diffusionPatterns.IsKey(datum.index)) parameterGrad.diffusionPatterns.AddDat(datum.index, diffusionPatternGradient);
      else parameterGrad.diffusionPatterns.GetDat(datum.index) += diffusionPatternGradient;
   
      delete[] srcNIds;
      delete[] dstNIds;
      delete[] vals;
      delete[] diffusionPatternVals;
   
      //printf("index:%d, k:%d, latent distribution:%f\n",datum.index(), key(), latentDistributions.GetDat(datum.index).GetDat(key)());     
      parameterGrad.kPi.GetDat(key) = latentDistributions.GetDat(datum.index).GetDat(key);
      parameterGrad.kPi_times.GetDat(key)++; 
   }
   return parameterGrad;
}

void MMRateFunction::maximize() {
   for (THash<TInt,TFlt>::TIter PI = parameter.kPi_times.BegI(); !PI.IsEnd(); PI++) {
      PI.GetDat() = 0.0;
   }
}

void MMRateFunction::set(MMRateFunctionConfigure configure) {
   shapingFunction = configure.shapingFunction;
   parameter.set(configure);
   parameterGrad.set(configure);
}

void MMRateParameter::set(MMRateFunctionConfigure configure) {
   Regularizer = configure.Regularizer;
   Mu = configure.Mu;
   Tol = configure.Tol;
   InitAlpha = configure.InitAlpha;
   MaxAlpha = configure.MaxAlpha;
   MinAlpha = configure.MinAlpha;
   InitDiffusionPattern = configure.InitDiffusionPattern;
   MaxDiffusionPattern = configure.MaxDiffusionPattern;
   MinDiffusionPattern = configure.MinDiffusionPattern;
   latentVariableSize = configure.latentVariableSize;

   for (TInt i=0;i<latentVariableSize;i++) {
      kAlphas.AddDat(i, THash<TIntPr, TFlt>());
      kPi.AddDat(i,TFlt::GetRnd());
      kPi_times.AddDat(i,0.0);
   }
   TFlt sum = 0.0;
   for (TInt i=0;i<latentVariableSize;i++) sum += kPi.GetDat(i);
   for (TInt i=0;i<latentVariableSize;i++) kPi.GetDat(i) /= sum;
}

void MMRateParameter::reset() {
   diffusionPatterns.Clr();
   for (THash<TInt, THash<TIntPr,TFlt> >::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      AI.GetDat().Clr();
   }
   for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) { 
      piI.GetDat() = 0.0;
      kPi_times.GetDat(piI.GetKey()) = 0.0;
   }
}

MMRateParameter& MMRateParameter::operator = (const MMRateParameter& p) {
   kPi.Clr();
   kPi = p.kPi;

   kPi_times.Clr();
   kPi_times = p.kPi_times;

   kAlphas.Clr();
   kAlphas = p.kAlphas;

   diffusionPatterns.Clr();
   diffusionPatterns = p.diffusionPatterns;
   
   return *this;
}

MMRateParameter& MMRateParameter::operator += (const MMRateParameter& p) {
   for (THash<TInt,TFlt>::TIter DI = p.diffusionPatterns.BegI(); !DI.IsEnd(); DI++) {
      TInt key = DI.GetKey();
      TFlt diffusionPattern = DI.GetDat();
      if (!diffusionPatterns.IsKey(key)) diffusionPatterns.AddDat(key, diffusionPattern);
      else diffusionPatterns.GetDat(key) += diffusionPattern;
   }
   for(THash<TInt, THash<TIntPr, TFlt> >::TIter AI = p.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      if (!kAlphas.IsKey(key)) {
         kAlphas.AddDat(key, THash<TIntPr, TFlt>());
         kPi.AddDat(key, 0.0);
         kPi_times.AddDat(key, 0.0);
      }

      THash<TIntPr, TFlt>& alphas = kAlphas.GetDat(key);
      for (THash<TIntPr,TFlt>::TIter aI = AI.GetDat().BegI(); !aI.IsEnd(); aI++) {
         TIntPr alphaIndex = aI.GetKey();
         TFlt alpha = aI.GetDat();
         if (!alphas.IsKey(alphaIndex)) alphas.AddDat(alphaIndex, alpha);
         else alphas.GetDat(alphaIndex) += alpha;
         //printf("topic: %d, %d,%d: += alpha:%f\n", key(), alphaIndex.Val1(), alphaIndex.Val2(), aI.GetDat());
      }
      
      kPi.GetDat(key) += p.kPi.GetDat(key);
      kPi_times.GetDat(key) += p.kPi_times.GetDat(key);
   }
   return *this;
}

MMRateParameter& MMRateParameter::operator *= (const TFlt multiplier) {
   for(THash<TInt, THash<TIntPr,TFlt> >::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      THash<TIntPr, TFlt>& alphas = AI.GetDat();
      for (THash<TIntPr,TFlt>::TIter aI = alphas.BegI(); !aI.IsEnd(); aI++) aI.GetDat() *= multiplier;
   }
   for (THash<TInt,TFlt>::TIter DI = diffusionPatterns.BegI(); !DI.IsEnd(); DI++) {
      DI.GetDat() *= multiplier;
   }
   return *this;
}

MMRateParameter& MMRateParameter::projectedlyUpdateGradient(const MMRateParameter& p) {
   for (THash<TInt,TFlt>::TIter DI = p.diffusionPatterns.BegI(); !DI.IsEnd(); DI++) {
      TInt key = DI.GetKey();
      TFlt diffusionPatternGradient = DI.GetDat(), diffusionPattern;
      if (diffusionPatterns.IsKey(key)) diffusionPattern = diffusionPatterns.GetDat(key); 
      else diffusionPattern = InitDiffusionPattern;

      diffusionPattern -= (diffusionPatternGradient + (Regularizer ? Mu : TFlt(0.0)) * diffusionPattern);

      if (diffusionPattern < MinDiffusionPattern) diffusionPattern = MinDiffusionPattern;
      if (diffusionPattern > MaxDiffusionPattern) diffusionPattern = MaxDiffusionPattern;

      if (!diffusionPatterns.IsKey(key)) diffusionPatterns.AddDat(key,diffusionPattern);
      else diffusionPatterns.GetDat(key) = diffusionPattern;
   }
   for(THash<TInt, THash<TIntPr,TFlt> >::TIter AI = p.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      THash<TIntPr,TFlt>& alphas = kAlphas.GetDat(key);
      for (THash<TIntPr,TFlt>::TIter aI = AI.GetDat().BegI(); !aI.IsEnd(); aI++) {
         TIntPr alphaIndex = aI.GetKey();
         TFlt alphaGradient = aI.GetDat(), alpha;
         if (alphas.IsKey(alphaIndex)) alpha = alphas.GetDat(alphaIndex); 
         else alpha = InitAlpha;

         alpha -= (alphaGradient + (Regularizer ? Mu : TFlt(0.0)) * alpha);

         if (alpha < Tol) alpha = Tol;
         if (alpha > MaxAlpha) alpha = MaxAlpha;

         if (!alphas.IsKey(alphaIndex)) alphas.AddDat(alphaIndex, alpha);
         else alphas.GetDat(alphaIndex) = alpha;
      }

      TFlt old = kPi.GetDat(key) * kPi_times.GetDat(key);
      kPi_times.GetDat(key) += p.kPi_times.GetDat(key);
      kPi.GetDat(key) = (old + p.kPi.GetDat(key))/kPi_times.GetDat(key);
   }
   return *this;
}
