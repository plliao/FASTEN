#include <AdditiveRiskFunction.h>

void AdditiveRiskFunction::set(AdditiveRiskFunctionConfigure configure) {
   shapingFunction = configure.shapingFunction;
   observedWindow = configure.observedWindow;
   parameter.set(configure);
}

AdditiveRiskParameter& AdditiveRiskFunction::gradient(Datum datum) {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;

   parameterGrad.reset();

   int nodeSize = NodeNmH.Len();
   int cascadeSize = Cascade.Len();
   int *srcNIds = new int[nodeSize * cascadeSize];
   int *dstNIds = new int[nodeSize * cascadeSize];
   float *vals  = new float[nodeSize * cascadeSize];
 
   for (int i=0;i<nodeSize;i++) {
      for (int j=0;j<cascadeSize;j++) {
         int index = i*cascadeSize + j;
         srcNIds[index] = dstNIds[index] = -1;
         vals[index] = -1.0;
      }
   }

   #pragma omp parallel for
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      THash<TInt, TNodeInfo>::TIter NI = NodeNmH.GetI(key);
      TInt dstNId = NI.GetKey(), srcNId;
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
            if (parameter.alphas.IsKey(alphaIndex)) alpha = parameter.alphas.GetDat(alphaIndex);
            else alpha = parameter.InitAlpha;
         
            sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
            //printf("sumInLog:%f, alpha:%f, val:%f, initAlpha:%f\n",sumInLog(),alpha(),shapingFunction->Value(srcTime,dstTime)(),parameter.InitAlpha());
         }
      }
      else dstTime = Cascade.GetMaxTm() + observedWindow; 

      int j=0;
      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++,j++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
         TIntPr key(srcNId, dstNId);
         if (!potentialEdges.IsKey(key)) continue;
                        
         TIntPr alphaIndex; alphaIndex.Val1 = srcNId; alphaIndex.Val2 = dstNId;
         if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime)
            val = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/sumInLog;
         else {
            val = shapingFunction->Integral(srcTime,dstTime);
         }
            
         int index = i*cascadeSize + j;
         srcNIds[index] = srcNId();
         dstNIds[index] = dstNId();
         vals[index] = val();
         //printf("index:%d, %d,%d: gradient:%f, shapingVal:%f, sumInLog:%f\n",datum.index(),srcNId(),dstNId(),val(),shapingFunction->Integral(srcTime,dstTime)(),sumInLog()); 
      }
   }

   for (int i=0;i<nodeSize;i++) {
      for (int j=0;j<cascadeSize;j++) {
         int index = i*cascadeSize + j;
         if (srcNIds[index]==-1) break;
         TIntPr alphaIndex; alphaIndex.Val1 = srcNIds[index]; alphaIndex.Val2 = dstNIds[index];
         parameterGrad.alphas.AddDat(alphaIndex,vals[index]);
      }
   }

   delete[] srcNIds;
   delete[] dstNIds;
   delete[] vals;

   return parameterGrad;
}

TFlt AdditiveRiskFunction::loss(Datum datum) const {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
   TFlt totalLoss = 0.0;

   int nodeSize = NodeNmH.Len();
   float *lossTable = new float[nodeSize];
   #pragma omp parallel for
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      THash<TInt, TNodeInfo>::TIter NI = NodeNmH.GetI(key);
      TInt dstNId = NI.GetKey(), srcNId;
      TFlt sumInLog = 0.0, val = 0.0;
      TFlt dstTime, srcTime;
      lossTable[i] = 0.0;

      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) dstTime = Cascade.GetTm(dstNId);
      else dstTime = Cascade.GetMaxTm() + observedWindow; 

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
                        
         TIntPr alphaIndex; alphaIndex.Val1 = srcNId; alphaIndex.Val2 = dstNId;

         TFlt alpha = 0.0;
         if (potentialEdges.IsKey(alphaIndex)) {
            if (parameter.alphas.IsKey(alphaIndex)) alpha = parameter.alphas.GetDat(alphaIndex);
            else alpha = parameter.InitAlpha;
         }

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
   return totalLoss;
}

void AdditiveRiskFunction::initPotentialEdges(Data data) {
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

AdditiveRiskParameter::AdditiveRiskParameter() {
   reset();
}

AdditiveRiskParameter& AdditiveRiskParameter::operator = (const AdditiveRiskParameter& p) {
   alphas.Clr();
   alphas = p.alphas;
   return *this; 
}

AdditiveRiskParameter& AdditiveRiskParameter::operator += (const AdditiveRiskParameter& p) {
   for (THash<TIntPr,TFlt>::TIter AI = p.alphas.BegI(); !AI.IsEnd(); AI++) {
      TIntPr key = AI.GetKey();
      TFlt alpha = AI.GetDat();
      if (!alphas.IsKey(key)) alphas.AddDat(key,alpha);
      else alphas.GetDat(key) += alpha;
      //printf("%d,%d: += alpha:%f \n", AI.GetDat()key.Val1(), key.Val2(), AI.GetDat());
   }
   return *this; 
}

AdditiveRiskParameter& AdditiveRiskParameter::operator *= (const TFlt multiplier) {
   for (THash<TIntPr,TFlt>::TIter AI = alphas.BegI(); !AI.IsEnd(); AI++) {
      AI.GetDat() *= multiplier;
   }
   return *this; 
}

AdditiveRiskParameter& AdditiveRiskParameter::projectedlyUpdateGradient(const AdditiveRiskParameter& p) {
   for (THash<TIntPr,TFlt>::TIter AI = p.alphas.BegI(); !AI.IsEnd(); AI++) {
      TIntPr key = AI.GetKey();
      TFlt alphaGradient = AI.GetDat(), alpha;
      //printf("%d,%d: %f, dat:%f, m:%f\n",key.Val1(),key.Val2(),alpha(),AI.GetDat()(),p.multiplier());
      if (alphas.IsKey(key)) alpha = alphas.GetDat(key); 
      else alpha = InitAlpha;

      alpha -= (alphaGradient + (Regularizer ? Mu : TFlt(0.0)) * alpha);

      if (alpha < Tol) alpha = Tol;
      if (alpha > MaxAlpha) alpha = MaxAlpha;

      if (!alphas.IsKey(key)) alphas.AddDat(key,alpha);
      else alphas.GetDat(key) = alpha;
   }
   return *this; 
}

void AdditiveRiskParameter::reset() {
   alphas.Clr();
}

void AdditiveRiskParameter::set(AdditiveRiskFunctionConfigure configure) {
   Regularizer = configure.Regularizer;
   Mu = configure.Mu;
   Tol = configure.Tol;
   InitAlpha = configure.InitAlpha;
   MaxAlpha = configure.MaxAlpha;
   MinAlpha = configure.MinAlpha;
}

