#ifndef PGD_H
#define PGD_H

#include <Parameter.h>
#include <cascdynetinf.h>
#include <InfoPathSampler.h>

template <typename T>
class PGDFunction;

struct PGDConfigure {
   size_t maxIterNm, batchSize;
   TFlt learningRate;
   TSampling sampling;
   TStr ParamSampling;
};

template <typename T>
class PGD {
   public:
      void set(PGDConfigure c) { 
         configure = c;
      }

      void Optimize(PGDFunction<T> &f, Data data) {
         iterNm = 0;
      
         double time = data.time;
         THash<TInt, TCascade> &cascH = data.cascH;
         TIntFltH &cascadesIdx = data.cascadesPositions;
         size_t scale = configure.maxIterNm / 5;
         TIntFltH sampledCascadesPositions;
         T learningRate;
      
         while(!IsTerminate()) { 
            T parameterDiff;
            for (size_t i=0;i<configure.batchSize;i++) {
               int index = InfoPathSampler::sample(configure.sampling, configure.ParamSampling, cascadesIdx.Len());
               sampledCascadesPositions.AddDat(cascadesIdx.GetKey(index), 0.0);
               Datum datum = {data.NodeNmH, cascH, cascH.GetKey(cascadesIdx.GetKey(index)), time};
               parameterDiff += f.gradient(datum);
            }
            f.calculateRMSProp(0.1, learningRate, parameterDiff);
            parameterDiff *= (configure.learningRate/double(configure.batchSize));
            f.parameter.projectedlyUpdateGradient(parameterDiff);
            iterNm++;
            if (iterNm % scale == 0) {
               double size = (double) sampledCascadesPositions.Len();
               Data sampleData = {data.NodeNmH, data.cascH, sampledCascadesPositions, data.time};
               loss = f.loss(sampleData)/size;
               printf("iterNm: %d, loss: %f\033[0K\r",(int)iterNm,loss());
               fflush(stdout);
            }
         }
         printf("\n");
      }

      bool IsTerminate() const {
         return iterNm >= configure.maxIterNm;
      }
   private:
      PGDConfigure configure;
      size_t iterNm;
      TFlt loss;
};

template<typename T> 
class PGDFunction {
   friend class PGD<T>;
   public:
      virtual T& gradient(Datum datum) = 0;
      virtual TFlt loss(Datum datum) const = 0;
      virtual void calculateRMSProp(TFlt, T&, T&) {}
      TFlt loss(Data data) const {
         TFlt totalLoss = 0.0;
         TIntFltH &cascadesPositions = data.cascadesPositions;
         for (TIntFltH::TIter CI = cascadesPositions.BegI(); !CI.IsEnd(); CI++) {
            TInt index = CI.GetKey();
            Datum datum = {data.NodeNmH, data.cascH, data.cascH.GetKey(index), data.time};
            totalLoss += loss(datum);
         } 
         return totalLoss;
      }
      const T& getParameter() const { return parameter;}
      const T& getParameterGrad() const { return parameterGrad;}
      T& getParameter() { return parameter;}
      T& getParameterGrad() { return parameterGrad;}
   public:
      T parameter, parameterGrad;
};

#endif
