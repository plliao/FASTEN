#include <MMRateFunction.h>

TFlt MMRateFunction::JointLikelihood(Datum datum, TInt latentVariable) const {
   TFlt logP = -1 * kAlphas.GetDat(latentVariable).loss(datum);
   TFlt logPi = TMath::Log(parameter.kPi.GetDat(latentVariable));
   //printf("logP: %f, logPi=%f\n",logP(),logPi());
   return logP + logPi;
}

MMRateParameter& MMRateFunction::gradient(Datum datum) {
   parameterGrad.reset();

   for (THash<TInt,AdditiveRiskFunction>::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      AdditiveRiskParameter& alphas = AI.GetDat().gradient(datum);
      alphas *= latentDistributions.GetDat(datum.index).GetDat(key);
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
   for (TInt i=0;i<configure.latentVariableSize;i++) {
      kAlphas.AddDat(i,AdditiveRiskFunction());
      kAlphas.GetDat(i).set(configure.configure);
   }
   parameter.init(configure.latentVariableSize, &kAlphas);
   parameterGrad.init(configure.latentVariableSize, &kAlphas);
}

void MMRateParameter::init(TInt latentVariableSize, THash<TInt,AdditiveRiskFunction>* KAlphasP) {
   kAlphasP = KAlphasP;
   for (TInt i=0;i<latentVariableSize;i++) {
      kPi.AddDat(i,TFlt::GetRnd());
      kPi_times.AddDat(i,0.0);
   }
   TFlt sum = 0.0;
   for (TInt i=0;i<latentVariableSize;i++) sum += kPi.GetDat(i);
   for (TInt i=0;i<latentVariableSize;i++) kPi.GetDat(i) /= sum;
}

void MMRateParameter::set(MMRateFunctionConfigure configure) {
   for (THash<TInt,AdditiveRiskFunction>::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      AdditiveRiskFunction& f = AI.GetDat();
      f.set(configure.configure);
   }
}

void MMRateParameter::reset() {
   for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) { 
      piI.GetDat() = 0.0;
      kPi_times.GetDat(piI.GetKey()) = 0.0;
   }
}

MMRateParameter& MMRateParameter::operator = (const MMRateParameter& p) {
   kAlphas.Clr();
   kPi.Clr();
   kPi_times.Clr();
   kAlphasP = p.kAlphasP;
   for(THash<TInt,AdditiveRiskFunction>::TIter AI = p.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      kAlphas.AddDat(key,AI.GetDat());   
      kPi.AddDat(key,p.kPi.GetDat(key));
      kPi_times.AddDat(key,p.kPi_times.GetDat(key));
   }
   return *this;
}

MMRateParameter& MMRateParameter::operator += (const MMRateParameter& p) {
   for(THash<TInt,AdditiveRiskFunction>::TIter AI = p.kAlphasP->BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      if (!kAlphas.IsKey(key)) {
         kAlphas.AddDat(key,AdditiveRiskFunction());
         kPi.AddDat(key,0.0);
         kPi_times.AddDat(key,0.0);
      }
      kAlphas.GetDat(key).getParameter() += AI.GetDat().getParameterGrad();
      kPi.GetDat(key) += p.kPi.GetDat(key);
      kPi_times.GetDat(key) += p.kPi_times.GetDat(key);
   }
   return *this;
}

MMRateParameter& MMRateParameter::operator *= (const TFlt multiplier) {
   for(THash<TInt,AdditiveRiskFunction>::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      AI.GetDat().getParameter() *= multiplier;
   }
   return *this;
}

MMRateParameter& MMRateParameter::projectedlyUpdateGradient(const MMRateParameter& p) {
   for(THash<TInt,AdditiveRiskFunction>::TIter AI = p.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      kAlphasP->GetDat(key).getParameter().projectedlyUpdateGradient(AI.GetDat().getParameter());

      TFlt old = kPi.GetDat(key) * kPi_times.GetDat(key);
      kPi_times.GetDat(key) += p.kPi_times.GetDat(key);
      kPi.GetDat(key) = (old + p.kPi.GetDat(key))/kPi_times.GetDat(key);
   }
   return *this;
}
