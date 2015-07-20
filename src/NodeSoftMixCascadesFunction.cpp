#include <NodeSoftMixCascadesFunction.h>

TFlt NodeSoftMixCascadesFunction::JointLikelihood(Datum datum, TInt latentVariable) const {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
   double totalLoss = 0.0;

   TInt startNId = Cascade.BegI().GetKey();

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

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;

         if (!shapingFunction->Before(srcTime,dstTime)) break; 
                        
         TFlt alpha = GetTopicAlpha(srcNId, dstNId, latentVariable);

         sumInLog += alpha * shapingFunction->Value(srcTime,dstTime);
         val += alpha * shapingFunction->Integral(srcTime,dstTime);
         //printf("sumInLog:%f,val:%f, alpha:%f, shapingVal:%f, shapingInt:%f\n",sumInLog(),val(),alpha(),shapingFunction->Value(srcTime,dstTime)(),shapingFunction->Integral(srcTime,dstTime)()); 
      }
      lossValue = val;
      
      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime && sumInLog!=0.0) lossValue -= TMath::Log(sumInLog);
      totalLoss += lossValue;
   }

   TFlt logPi = TMath::Log(parameter.nodeWeights.GetDat(startNId).GetDat(latentVariable));
   //printf("datum:%d, Myloss:%f, logPi:%f\n",datum.index(), totalLoss, logPi());
   return logPi - totalLoss;
}

NodeSoftMixCascadesParameter& NodeSoftMixCascadesFunction::gradient(Datum datum) {
   double CurrentTime = datum.time;
   TCascade &Cascade = datum.cascH.GetDat(datum.index);
   THash<TInt, TNodeInfo> &NodeNmH = datum.NodeNmH;
 
   parameterGrad.reset();
   TInt startNId = Cascade.BegI().GetKey();
   if (!parameterGrad.nodeWeights.IsKey(startNId)) {
      parameterGrad.nodeWeights.AddDat(startNId, THash<TInt,TFlt>());
      parameterGrad.nodeSampledTimes.AddDat(startNId, 0.0);
      THash<TInt, TFlt>& weight = parameterGrad.nodeWeights.GetDat(startNId);
      for (TInt i = 0; i < parameter.latentVariableSize; i++) {
         weight.AddDat(i, 0.0);
      }
   }

   THash<TInt, TFlt>& weight = parameterGrad.nodeWeights.GetDat(startNId);
   for (TInt i = 0; i < parameter.latentVariableSize; i++) {
      parameterGrad.kAlphas.AddDat(i, THash<TIntPr, TFlt>());
      weight.GetDat(i) += latentDistributions.GetDat(datum.index).GetDat(i);
   }
   parameterGrad.nodeSampledTimes.GetDat(startNId)++;

   int nodeSize = NodeNmH.Len();
   #pragma omp parallel for
   for (int i=0; i<nodeSize; i++) {
      TInt dstNId = NodeNmH.GetKey(i), srcNId;
      TFlt dstTime, srcTime;
      THash<TInt,TFlt> dstAlphas;

      for (TInt i = 0; i < parameter.latentVariableSize; i++) dstAlphas.AddDat(i, 0.0);

      if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime) {
         dstTime = Cascade.GetTm(dstNId);
         for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
            srcNId = CascadeNI.GetKey();
            srcTime = CascadeNI.GetDat().Tm;

            if (!shapingFunction->Before(srcTime,dstTime)) break; 
                         
            for (TInt i = 0; i < parameter.latentVariableSize; i++) {
               TFlt alpha = parameter.GetTopicAlpha(srcNId, dstNId, i);
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

      for (THash<TInt, THitInfo>::TIter CascadeNI = Cascade.BegI(); CascadeNI < Cascade.EndI(); CascadeNI++) {
         srcNId = CascadeNI.GetKey();
         srcTime = CascadeNI.GetDat().Tm;
   
         if (!shapingFunction->Before(srcTime,dstTime)) break; 
                           
         TIntPr alphaIndex; alphaIndex.Val1 = srcNId; alphaIndex.Val2 = dstNId;
         TFlt val = 0.0;

         for (TInt i = 0; i < parameter.latentVariableSize; i++) {
            if (Cascade.IsNode(dstNId) && Cascade.GetTm(dstNId) <= CurrentTime)
               val = shapingFunction->Integral(srcTime,dstTime) - shapingFunction->Value(srcTime,dstTime) / dstAlphas.GetDat(i);
            else
               val = shapingFunction->Integral(srcTime,dstTime);
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


static void updateRMSProp(TFlt alpha, THash<TIntPr,TFlt>& lr, THash<TIntPr,TFlt>& gradient) {
   for(THash<TIntPr,TFlt>::TIter GI = gradient.BegI(); !GI.IsEnd(); GI++) {
      TIntPr key = GI.GetKey();
      if (!lr.IsKey(key)) lr.AddDat(key, TMath::Sqrt(GI.GetDat() * GI.GetDat()));
      else lr.GetDat(key) = TMath::Sqrt(alpha * lr.GetDat(key) * lr.GetDat(key) + (1.0 - alpha) * GI.GetDat() * GI.GetDat());
      GI.GetDat() /= lr.GetDat(key);
   }
}

static void updateRMSProp(TFlt alpha, THash<TInt,TFlt>& lr, THash<TInt,TFlt>& gradient) {
   for(THash<TInt,TFlt>::TIter GI = gradient.BegI(); !GI.IsEnd(); GI++) {
      TInt key = GI.GetKey();
      if (!lr.IsKey(key)) lr.AddDat(key, TMath::Sqrt(GI.GetDat() * GI.GetDat()));
      else lr.GetDat(key) = TMath::Sqrt(alpha * lr.GetDat(key) * lr.GetDat(key) + (1.0 - alpha) * GI.GetDat() * GI.GetDat());
      GI.GetDat() /= lr.GetDat(key);
   }
}

void NodeSoftMixCascadesFunction::calculateRMSProp(TFlt alpha, NodeSoftMixCascadesParameter& lr, NodeSoftMixCascadesParameter& gradient) {
  for (THash<TInt, THash<TIntPr,TFlt> >::TIter AI = gradient.kAlphas.BegI(); !AI.IsEnd(); AI++) {
     if (!lr.kAlphas.IsKey(AI.GetKey())) lr.kAlphas.AddDat(AI.GetKey(), THash<TIntPr,TFlt>());
     updateRMSProp(alpha, lr.kAlphas.GetDat(AI.GetKey()), AI.GetDat());
  }
}

void NodeSoftMixCascadesFunction::maximize() {
   for (THash<TInt,THash<TInt,TFlt> >::TIter WI = parameterGrad.nodeWeights.BegI(); !WI.IsEnd(); WI++) {
      THash<TInt,TFlt>& weight = parameter.nodeWeights.GetDat(WI.GetKey());
      TFlt& times = parameterGrad.nodeSampledTimes.GetDat(WI.GetKey());
      if (times == 0.0) continue;
      //printf("node %d, times %f, ", WI.GetKey()(), times());
      for (THash<TInt,TFlt>::TIter VI = WI.GetDat().BegI(); !VI.IsEnd(); VI++) {
         //printf("topic %d, value %f, ", VI.GetKey()(), VI.GetDat()());
         weight.GetDat(VI.GetKey()) = VI.GetDat() / times;
         if (weight.GetDat(VI.GetKey()) < 0.001) weight.GetDat(VI.GetKey()) = 0.001; 
         VI.GetDat() = 0.0;
         //printf(" weight %f\t", weight.GetDat(VI.GetKey())());
      }
      //printf("\n");
      times = 0.0;
   }
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
}

NodeSoftMixCascadesParameter& NodeSoftMixCascadesParameter::operator = (const NodeSoftMixCascadesParameter& p) {

   kAlphas.Clr();
   kAlphas = p.kAlphas;

   nodeWeights.Clr();
   nodeWeights = p.nodeWeights;

   nodeSampledTimes.Clr();
   nodeSampledTimes = p.nodeSampledTimes; 
   return *this;
}

NodeSoftMixCascadesParameter& NodeSoftMixCascadesParameter::operator += (const NodeSoftMixCascadesParameter& p) {
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
   return *this;
}

TFlt NodeSoftMixCascadesParameter::GetTopicAlpha(TInt srcNId, TInt dstNId, TInt topic) const {
   const THash<TIntPr, TFlt>& alphas = kAlphas.GetDat(topic);
   TIntPr index(srcNId,dstNId);
   if (alphas.IsKey(index)) return alphas.GetDat(index);
   return InitAlpha;
}

TFlt NodeSoftMixCascadesParameter::GetAlpha(TInt srcNId, TInt dstNId, TInt topic) const {
  const THash<TIntPr,TFlt>& alphas = kAlphas.GetDat(topic);
  TFlt alpha = 0.0;
  TIntPr index(srcNId,dstNId);
  if (!alphas.IsKey(index)) return alpha;
  return alphas.GetDat(index); 
}
