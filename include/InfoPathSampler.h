#ifndef INFOPATHSAMPLER_H
#define INFOPATHSAMPLER_H

#include <cascdynetinf.h>

class InfoPathSampler {
   public:
      static int sample(const TSampling& Sampling, const TStr& ParamSampling, const int range);
};

#endif
