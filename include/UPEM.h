#ifndef UPEM_H
#define UPEM_H

#include <Parameter.h>
#include <PGD.h>
#include <cascdynetinf.h>

template <typename parameter>
class UPEMLikelihoodFunction;

typedef struct {
   PGDConfigure pGDConfigure;
   size_t maxIterNm, maxCoorIterNm;
   TInt latentVariableSize;
}UPEMConfigure;


template<typename parameter>
class UPEM {
   public:
      void Optimize(UPEMLikelihoodFunction<parameter> &LF, Data data) {
         emIterNm = 0;
         while(!IsTerminate()) {
            Expectation(LF,data);      
            Maximization(LF,data);
            emIterNm++;

            THash<TInt,TFlt> kPi = LF.getPriorTHash();
            printf("UPEM iteration:%d, ",(int)emIterNm);
            for (int i=0; i<kPi.Len(); i++) {
               TInt key; TFlt value;
               kPi.GetKeyDat(i,key,value);
               printf("topic %d: %f, ", key(), value());
            }
            printf("\n");
            fflush(stdout);
         }
      }
      bool IsTerminate() const {
         return emIterNm >= configure.maxIterNm; 
      }
      void set(UPEMConfigure configure) {
         this->configure = configure;;
      }

   private:
      UPEMConfigure configure;
      size_t iterNm, coorIterNm, emIterNm;
      TFlt loss;

      void Expectation(UPEMLikelihoodFunction<parameter> &LF, Data data) const {
         for (THash<TInt, TCascade>::TIter CI = data.cascH.BegI(); !CI.IsEnd(); CI++) {
            Datum datum = {data.NodeNmH, data.cascH, CI.GetKey(), data.time};

            TInt size = configure.latentVariableSize;
            THash<TInt,TFlt> jointLikelihoodTable;
            for (TInt latentVariable=0; latentVariable < size; latentVariable++) {
               jointLikelihoodTable.AddDat(latentVariable, LF.JointLikelihood(datum,latentVariable));
            }

            //printf("\nindex:%d",CI.GetKey()());
            THash<TInt,TFlt> &latentDistribution = LF.latentDistributions.GetDat(CI.GetKey());
            for (TInt latentVariable=0; latentVariable < size; latentVariable++) {
               TFlt likelihood = 0.0;
               for (TInt i=0; i < size; i++)
                  likelihood += TMath::Power(TMath::E, jointLikelihoodTable.GetDat(i) - jointLikelihoodTable.GetDat(latentVariable));
               latentDistribution.GetDat(latentVariable) = 1.0/likelihood;
               //printf(", k:%d, p:%f",latentVariable(),latentDistribution.GetDat(latentVariable)());
            }
            //printf("\n\n");
         }
      }
      void Maximization(UPEMLikelihoodFunction<parameter> &LF, Data data) {
         
         coorIterNm = 0;
      
         double time = data.time;
         THash<TInt, TCascade> &cascH = data.cascH;
         TIntFltH &cascadesIdx = data.cascadesIdx;
      
         while(coorIterNm < configure.maxCoorIterNm) {
            //maximize acquaintance
            iterNm = 0;
            while(iterNm < configure.pGDConfigure.maxIterNm) { 
               parameter parameterDiff;
               for (size_t i=0;i<configure.pGDConfigure.batchSize;i++) {
                  int index = InfoPathSampler::sample(configure.pGDConfigure.sampling, configure.pGDConfigure.ParamSampling, cascadesIdx.Len());
                  Datum datum = {data.NodeNmH, cascH, cascH.GetKey(cascadesIdx.GetKey(index)), time};
                  parameterDiff += LF.gradient3(datum);
               }
               parameterDiff *= (configure.pGDConfigure.learningRate/double(configure.pGDConfigure.batchSize));
               LF.parameter.projectedlyUpdateGradient(parameterDiff);
               iterNm++;
               //printf("iterNm: %d, loss: %f\n",(int)iterNm,loss());
            }

            //maximize receiver
            iterNm = 0;
            while(iterNm < configure.pGDConfigure.maxIterNm) { 
               parameter parameterDiff;
               for (size_t i=0;i<configure.pGDConfigure.batchSize;i++) {
                  int index = InfoPathSampler::sample(configure.pGDConfigure.sampling, configure.pGDConfigure.ParamSampling, cascadesIdx.Len());
                  Datum datum = {data.NodeNmH, cascH, cascH.GetKey(cascadesIdx.GetKey(index)), time};
                  parameterDiff += LF.gradient1(datum);
               }
               parameterDiff *= (configure.pGDConfigure.learningRate/double(configure.pGDConfigure.batchSize));
               LF.parameter.projectedlyUpdateGradient(parameterDiff);
               iterNm++;
               //printf("iterNm: %d, loss: %f\n",(int)iterNm,loss());
            }

            //maximize spreader
            /*iterNm = 0;
            while(iterNm < configure.pGDConfigure.maxIterNm) { 
               parameter parameterDiff;
               for (size_t i=0;i<configure.pGDConfigure.batchSize;i++) {
                  int index = InfoPathSampler::sample(configure.pGDConfigure.sampling, configure.pGDConfigure.ParamSampling, cascadesIdx.Len());
                  Datum datum = {data.NodeNmH, cascH, cascH.GetKey(cascadesIdx.GetKey(index)), time};
                  parameterDiff += LF.gradient2(datum);
               }
               parameterDiff *= (0.1 * configure.pGDConfigure.learningRate/double(configure.pGDConfigure.batchSize));
               LF.parameter.projectedlyUpdateGradient(parameterDiff);
               iterNm++;
               //printf("iterNm: %d, loss: %f\n",(int)iterNm,loss());
            }*/
            coorIterNm++;
            printf("COOR iteration:%d\n",(int)coorIterNm);
            fflush(stdout);
         }
         LF.maximize(); 
      }
};

template<typename parameter>
class UPEMLikelihoodFunction : public PGDFunction<parameter> {
   friend class UPEM<parameter>;
   public:
      virtual TFlt JointLikelihood(Datum datum, TInt latentVariable) const = 0;
      virtual void maximize() = 0;
      virtual parameter& gradient1(Datum datum) = 0;
      virtual parameter& gradient2(Datum datum) = 0;
      virtual parameter& gradient3(Datum datum) = 0;
      virtual THash<TInt,TFlt> getPriorTHash() const = 0;
      TFlt loss(Datum datum) const {
         TFlt datumLoss = 0.0;
         for (TInt i=0;i<latentVariableSize;i++) datumLoss += TMath::Power(TMath::E, JointLikelihood(datum,i));
         return datumLoss;
      }
      void InitLatentVariable(Data data, UPEMConfigure configure) {
         latentDistributions.Clr();
         latentVariableSize = configure.latentVariableSize;
         for (THash<TInt, TCascade>::TIter CI = data.cascH.BegI(); !CI.IsEnd(); CI++) {
            THash<TInt,TFlt> latentDistribution;
            for (TInt i=0;i<latentVariableSize;i++) latentDistribution.AddDat(i,double(1/latentVariableSize));
            latentDistributions.AddDat(CI.GetKey(),latentDistribution);
         }
      }
   protected:
      TInt latentVariableSize;
      THash<TInt, THash<TInt,TFlt> > latentDistributions;
};

#endif
