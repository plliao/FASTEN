#include <InfoPathSampler.h>

int InfoPathSampler::sample(const TSampling& Sampling, const TStr& ParamSampling, const int range) {

   int sampleValue = -1;
   TStrV ParamSamplingV; ParamSampling.SplitOnAllCh(';', ParamSamplingV);

   switch (Sampling) {
     case UNIF_SAMPLING:
       sampleValue = TInt::Rnd.GetUniDevInt(range);
       break;

     case WIN_SAMPLING:
       sampleValue = TInt::Rnd.GetUniDevInt(range);
       break;

     case EXP_SAMPLING:
       do {
         sampleValue = (int)TFlt::Rnd.GetExpDev(ParamSamplingV[0].GetFlt());
       } while (sampleValue > range-1);
       break;

     case WIN_EXP_SAMPLING:
       do {
         sampleValue = (int)TFlt::Rnd.GetExpDev(ParamSamplingV[1].GetFlt());
       } while (sampleValue > range-1);
       break;

     case RAY_SAMPLING:
       do {
         sampleValue = (int)TFlt::Rnd.GetRayleigh(ParamSamplingV[0].GetFlt());
       } while (sampleValue > range-1);
       break;
   }
   return sampleValue;
}
