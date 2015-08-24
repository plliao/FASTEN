#include <MixCascadesFunction.h>

TFlt MixCascadesFunction::JointLikelihood(Datum datum, TInt latentVariable) const {
   TFlt logP = -1 * parameter.kAlphas.GetDat(latentVariable).loss(datum);
   TFlt logPi = TMath::Log(parameter.kPi.GetDat(latentVariable));
   //printf("logP: %f, logPi=%f\n",logP(),logPi());
   return logP + logPi;
}

MixCascadesParameter& MixCascadesFunction::gradient(Datum datum) {
   parameter.reset();

   for (THash<TInt,AdditiveRiskFunction>::TIter AI = parameter.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      AdditiveRiskParameter& alphas = AI.GetDat().gradient(datum);
      alphas *= latentDistributions.GetDat(datum.index).GetDat(key);
      //printf("index:%d, k:%d, latent distribution:%f\n",datum.index(), key(), latentDistributions.GetDat(datum.index).GetDat(key)());     
      parameterGrad.kPi.GetDat(key) += latentDistributions.GetDat(datum.index).GetDat(key);
      parameterGrad.kPi_times.GetDat(key)++; 
   }
   return parameter;
}

void MixCascadesFunction::maximize() {
   for (THash<TInt,TFlt>::TIter PI = parameterGrad.kPi_times.BegI(); !PI.IsEnd(); PI++) {
      if (PI.GetDat()!=0.0) {
         parameter.kPi.GetDat(PI.GetKey()) = parameterGrad.kPi.GetDat(PI.GetKey()) / PI.GetDat();
      }
      parameterGrad.kPi.GetDat(PI.GetKey()) = 0.0;
      PI.GetDat() = 0.0;
   }
}

void MixCascadesFunction::initPotentialEdges(Data data) {
  for (THash<TInt,AdditiveRiskFunction>::TIter AI = parameter.kAlphas.BegI(); !AI.IsEnd(); AI++) {
     AI.GetDat().initPotentialEdges(data);
  }
}

void MixCascadesFunction::init(TInt latentVariableSize) {
   parameter.init(latentVariableSize);
   parameterGrad.init(latentVariableSize);
}

void MixCascadesFunction::set(MixCascadesFunctionConfigure configure) {
   latentVariableSize = configure.latentVariableSize;
   parameter.set(configure);
}

void MixCascadesFunction::initKPiParameter() {
   parameter.initKPiParameter();
}

void MixCascadesParameter::set(MixCascadesFunctionConfigure configure) {
   for (THash<TInt,AdditiveRiskFunction>::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      AI.GetDat().set(configure.configure);
   }
}

void MixCascadesParameter::init(TInt latentVariableSize) {
   for (TInt i=0; i<latentVariableSize; i++) {
      kAlphas.AddDat(i,AdditiveRiskFunction());
      kPi.AddDat(i, 1.0 / double(latentVariableSize));
      kPi_times.AddDat(i, 0.0);
   }
}

void MixCascadesParameter::initKPiParameter() {
   TRnd rnd; rnd.PutSeed(time(NULL));
   for (THash<TInt,TFlt>::TIter PI = kPi.BegI(); !PI.IsEnd(); PI++) 
      PI.GetDat() =  rnd.GetUniDev() * 1.0 + 1.0;
   
   TFlt sum = 0.0;
   for (THash<TInt,TFlt>::TIter PI = kPi.BegI(); !PI.IsEnd(); PI++) sum += PI.GetDat();
   for (THash<TInt,TFlt>::TIter PI = kPi.BegI(); !PI.IsEnd(); PI++) PI.GetDat() /= sum;
}


void MixCascadesParameter::reset() {
   for (THash<TInt,AdditiveRiskFunction>::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      AI.GetDat().parameterGrad.reset();
   }
}

MixCascadesParameter& MixCascadesParameter::operator = (const MixCascadesParameter& p) {
   kAlphas.Clr();
   kPi.Clr();
   kPi_times.Clr();

   kAlphas = p.kAlphas;
   kPi = p.kPi;
   kPi_times = p.kPi_times;
   return *this;
}

MixCascadesParameter& MixCascadesParameter::operator += (const MixCascadesParameter& p) {
   for(THash<TInt,AdditiveRiskFunction>::TIter AI = p.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      if (!kAlphas.IsKey(key)) {
         kAlphas.AddDat(key,AdditiveRiskFunction());
      }
      kAlphas.GetDat(key).parameter += AI.GetDat().parameterGrad;
   }
   return *this;
}

MixCascadesParameter& MixCascadesParameter::operator *= (const TFlt multiplier) {
   for(THash<TInt,AdditiveRiskFunction>::TIter AI = kAlphas.BegI(); !AI.IsEnd(); AI++) {
      AI.GetDat().parameter *= multiplier;
   }
   return *this;
}

MixCascadesParameter& MixCascadesParameter::projectedlyUpdateGradient(const MixCascadesParameter& p) {
   for(THash<TInt,AdditiveRiskFunction>::TIter AI = p.kAlphas.BegI(); !AI.IsEnd(); AI++) {
      TInt key = AI.GetKey();
      kAlphas.GetDat(key).parameter.projectedlyUpdateGradient(AI.GetDat().parameter);
   }
   return *this;
}
