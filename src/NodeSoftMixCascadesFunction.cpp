#include <NodeSoftMixCascadesFunction.h>

TFlt NodeSoftMixCascadesFunction::loss(Datum datum) const {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
   double totalLoss = 0.0;

   TInt startNId = Cascade.BegI().GetKey();

   int nodeSize = NodeNmH.Len();
   //float *lossTable = new float[nodeSize];
   #pragma omp parallel for reduction(+:totalLoss)
   for (int i=0;i<nodeSize;i++) {
      TInt key = NodeNmH.GetKey(i);
      TInt dstNId = key, srcNId;
      TFlt sumInLog = 0.0, val = 0.0;
      TFlt dstTime, srcTime;
      double lossValue;

      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) dstTime = Cascade.GetTm(dstNId);
      else dstTime = Cascade.GetMaxTm() + observedWindow;

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
                        
         TFlt alpha = GetAlpha(srcNId, dstNId, startNId);

         sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
         val += alpha * shapingFunction->Integral(srcTime,dstTime);
         //printf("sumInLog:%f,val:%f, alpha:%f, shapingVal:%f, shapingInt:%f\n",sumInLog(),val(),alpha(),shapingFunction->Value(srcTime,dstTime)(),shapingFunction->Integral(srcTime,dstTime)()); 
      }
      lossValue = val;
      
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime && sumInLog!=0.0) lossValue -= TMath::Log(sumInLog);
      totalLoss += lossValue;
   }

   //for (int i=0;i<nodeSize;i++) totalLoss += lossTable[i];
   //delete[] lossTable;

   //printf("datum:%d, Myloss:%f\n",datum.index(), totalLoss());
   return -1.0 * totalLoss;
}

NodeSoftMixCascadesParameter& NodeSoftMixCascadesFunction::gradient(Datum datum) {

   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
   TInt startNId = Cascade.BegI().GetKey();
 
   THash<TInt, TFlt>& weight = parameter.nodeWeights.GetDat(startNId);
   int nodeSize = NodeNmH.Len();

   parameterGrad.reset();
   for (TInt i = 0; i < parameter.latentVariableSize; i++) {
      parameterGrad.kAlphas.AddDat(i, THash<TIntPr, TFlt>());
   }
   parameterGrad.nodeWeights.AddDat(startNId, THash<TInt, TFlt>());

   #pragma omp parallel for
   for (int i=0; i<nodeSize; i++) {
      TInt dstNId = NodeNmH.GetKey(i), srcNId;
      TFlt sumInLog = 0.0, val = 0.0;
      TFlt dstTime, srcTime;

      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) {
         dstTime = Cascade.GetTm(dstNId);
         for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
            srcNId = CascadeNI.GetKey();
            srcTime = CascadeNI.GetDat().Tm;

            if (!shapingFunction->Before(srcTime,dstTime)) break; 
                         
            TFlt alpha = 0.0;
            for (TInt i = 0; i < parameter.latentVariableSize; i++) {
               TFlt value = parameter.GetTopicAlpha(srcNId, dstNId, i);
               alpha += value * weight.GetDat(i);
            }
   
            sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
            //printf("sumInLog:%f, alpha:%f, val:%f, initAlpha:%f\n",sumInLog(),alpha(),shapingFunction->Value(srcTime,dstTime)(),parameter.InitAlpha());
         }
      }
      else dstTime = Cascade.GetMaxTm() + observedWindow;
   
      THash<TInt, TFlt> weightGradient;
      THash<TInt, THash<TIntPr, TFlt> > kAlphasGradient;
      for (TInt i = 0; i < parameter.latentVariableSize; i++) {
         weightGradient.AddDat(i, 0.0);
         kAlphasGradient.AddDat(i, THash<TIntPr, TFlt>());
      }

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;
   
         if (!shapingFunction->Before(srcTime,dstTime)) break; 
                           
         TIntPr alphaIndex; alphaIndex.Val1 = srcNId; alphaIndex.Val2 = dstNId;
         if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime)
            val = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime)/sumInLog;
         else
            val = shapingFunction->Integral(srcTime,dstTime);
         
         for (TInt i = 0; i < parameter.latentVariableSize; i++) {
            THash<TIntPr, TFlt>& alphaGradient = kAlphasGradient.GetDat(i);
            alphaGradient.AddDat(alphaIndex, val * weight.GetDat(i));
            weightGradient.GetDat(i) += val * parameter.GetTopicAlpha(srcNId, dstNId, i);
         }
         //printf("index:%d, %d,%d: gradient:%f, shapingVal:%f, sumInLog:%f\n",datum.index(),srcNId(),dstNId(),val(),shapingFunction->Integral(srcTime,dstTime)(),sumInLog()); 
      }

      #pragma omp critical 
      {
         THash<TInt, TFlt>& weight = parameterGrad.nodeWeights.GetDat(startNId);
         for (TInt i = 0; i < parameter.latentVariableSize; i++) {
            THash<TIntPr, TFlt>& alphaGradient = kAlphasGradient.GetDat(i);
            THash<TIntPr, TFlt>& alpha = parameterGrad.kAlphas.GetDat(i);
            for (THash<TIntPr, TFlt>::TIter AI = alphaGradient.BegI(); !AI.IsEnd(); AI++) {
               if (alpha.IsKey(AI.GetKey())) alpha.GetDat(AI.GetKey()) += AI.GetDat();
               else alpha.AddDat(AI.GetKey(), AI.GetDat());
            }

            if (weight.IsKey(i)) weight.GetDat(i) += weightGradient.GetDat(i);
            else weight.AddDat(i, weightGradient.GetDat(i));
         }
      }

   }
   
   return parameterGrad;
}

void NodeSoftMixCascadesFunction::set(NodeSoftMixCascadesFunctionConfigure configure) {
   shapingFunction = configure.shapingFunction;
   parameter.set(configure);
   parameterGrad.set(configure);
}

void NodeSoftMixCascadesFunction::init(Data data, TInt NodeNm) {
   parameter.init(data, NodeNm);
   //parameterGrad.init(data);
}

void NodeSoftMixCascadesParameter::set(NodeSoftMixCascadesFunctionConfigure configure) {
   Regularizer = configure.Regularizer;
   Mu = configure.Mu;
   Tol = configure.Tol;
   InitAlpha = configure.InitAlpha;
   MaxAlpha = configure.MaxAlpha;
   MinAlpha = configure.MinAlpha;
   latentVariableSize = configure.latentVariableSize;
}

void NodeSoftMixCascadesParameter::init(Data data, TInt NodeNm) {
   for (TInt i=0; i < latentVariableSize; i++) {
      kAlphas.AddDat(i, THash<TIntPr, TFlt>());
   }

   if (NodeNm==0) {
      for (THash<TInt, TNodeInfo>::TIter NI = data.NodeNmH.BegI(); !NI.IsEnd(); NI++) {
         nodeWeights.AddDat(NI.GetKey(), THash<TInt, TFlt>());
      }
   }
   else {
      for (TInt i=0; i<NodeNm; i++) {
         nodeWeights.AddDat(i, THash<TInt, TFlt>());
      }
   }
}

void NodeSoftMixCascadesParameter::initWeightParameter() {
   for (THash<TInt, THash<TInt, TFlt> >::TIter WI = nodeWeights.BegI(); !WI.IsEnd(); WI++) {
      THash<TInt, TFlt>& weight = WI.GetDat();
      TFlt sum = 0.0;
      for (TInt i=0; i < latentVariableSize; i++) {
         weight.AddDat(i, TFlt::Rnd.GetUniDev());
         sum += weight.GetDat(i);
      }
      for (TInt i=0; i < latentVariableSize; i++) weight.GetDat(i) = weight.GetDat(i) / sum;
   }
}

void NodeSoftMixCascadesParameter::initAlphaParameter() {
   for (TInt i=0; i < latentVariableSize; i++) {
      THash<TIntPr, TFlt>& alphas = kAlphas.GetDat(i);
      for (THash<TIntPr, TFlt>::TIter AI = alphas.BegI(); !AI.IsEnd(); AI++) {
         AI.GetDat() = TFlt::Rnd.GetUniDev() * (MaxAlpha - MinAlpha) + MinAlpha;
      }
   }
}

void NodeSoftMixCascadesParameter::reset() {
   kAlphas.Clr();
   nodeWeights.Clr();
}

NodeSoftMixCascadesParameter& NodeSoftMixCascadesParameter::operator = (const NodeSoftMixCascadesParameter& p) {

   kAlphas.Clr();
   kAlphas = p.kAlphas;

   nodeWeights.Clr();
   nodeWeights = p.nodeWeights;
   return *this;
}

NodeSoftMixCascadesParameter& NodeSoftMixCascadesParameter::operator += (const NodeSoftMixCascadesParameter& p) {
   for (THash<TInt,THash<TInt,TFlt> >::TIter CI = p.nodeWeights.BegI(); !CI.IsEnd(); CI++) {
      if (!nodeWeights.IsKey(CI.GetKey())) nodeWeights.AddDat(CI.GetKey(), THash<TInt,TFlt>());
      THash<TInt,TFlt>& weight = nodeWeights.GetDat(CI.GetKey());

      for (THash<TInt,TFlt>::TIter VI = CI.GetDat().BegI(); !VI.IsEnd(); VI++) {
         if (weight.IsKey(VI.GetKey())) weight.GetDat(VI.GetKey()) += VI.GetDat();
         else weight.AddDat(VI.GetKey(), VI.GetDat());
      }
   }
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

NodeSoftMixCascadesParameter& NodeSoftMixCascadesParameter::operator *= (const TFlt multiplier) {
   for(THash<TInt, THash<TIntPr,TFlt> >::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      THash<TIntPr, TFlt>& alphas = AI.GetDat();
      for (THash<TIntPr,TFlt>::TIter aI = alphas.BegI(); !aI.IsEnd(); aI++) aI.GetDat() *= multiplier;
   }
   for (THash<TInt,THash<TInt,TFlt> >::TIter CI = nodeWeights.BegI(); !CI.IsEnd(); CI++) {
      for (THash<TInt,TFlt>::TIter VI = CI.GetDat().BegI(); !VI.IsEnd(); VI++) VI.GetDat() *= multiplier;
   }
   return *this;
}

NodeSoftMixCascadesParameter& NodeSoftMixCascadesParameter::projectedlyUpdateGradient(const NodeSoftMixCascadesParameter& p) {
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

   for (THash<TInt,THash<TInt,TFlt> >::TIter CI = p.nodeWeights.BegI(); !CI.IsEnd(); CI++) {
      THash<TInt,TFlt>& weight = nodeWeights.GetDat(CI.GetKey());

      TFlt sum = 0.0, C = 0.5;;
      for (THash<TInt,TFlt>::TIter VI = weight.BegI(); !VI.IsEnd(); VI++) sum += VI.GetDat();

      for (THash<TInt,TFlt>::TIter VI = CI.GetDat().BegI(); !VI.IsEnd(); VI++) {

         TFlt value = weight.GetDat(VI.GetKey()), gradient = VI.GetDat() + C * (sum - 1.0);
         value -= gradient;
         
         if (value < 0.0) value = 0.001;
         if (value > 1.0) value = 1.000;

         weight.GetDat(VI.GetKey()) = value;
      }
   }
   return *this;
}

TFlt NodeSoftMixCascadesParameter::GetTopicAlpha(TInt srcNId, TInt dstNId, TInt topic) const {
   const THash<TIntPr, TFlt>& alphas = kAlphas.GetDat(topic);
   TIntPr index(srcNId,dstNId);
   if (alphas.IsKey(index)) return alphas.GetDat(index);
   return InitAlpha;
}

TFlt NodeSoftMixCascadesParameter::GetAlpha(TInt srcNId, TInt dstNId, TInt NId) const {
  if (!nodeWeights.IsKey(NId)) return 0.0;
  const THash<TInt, TFlt>& weight = nodeWeights.GetDat(NId);

  TFlt alpha = 0.0; 
  for (THash<TInt, THash<TIntPr,TFlt> >::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
     TIntPr index(srcNId,dstNId);
     if (AI.GetDat().IsKey(index)) alpha += AI.GetDat().GetDat(index) * weight.GetDat(AI.GetKey());
  }
  return alpha;
}
