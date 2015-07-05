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
   double initialMomentum, finalMomentum, momentumRatio;
}UPEMConfigure;


template<typename parameter>
class UPEM {
   public:
      void Optimize(UPEMLikelihoodFunction<parameter> &LF, Data data) {
         emIterNm = 0;

         sampledCascadesPositions.Reserve(configure.maxCoorIterNm * configure.pGDConfigure.maxIterNm * configure.pGDConfigure.batchSize);

         while(!IsTerminate()) {

            coorIterNm = 0;
            while(coorIterNm < configure.maxCoorIterNm) {
               iterNm = 0;
               while(iterNm < configure.pGDConfigure.maxIterNm) { 
                  for (size_t i=0;i<configure.pGDConfigure.batchSize;i++) {
                     int index = InfoPathSampler::sample(configure.pGDConfigure.sampling, configure.pGDConfigure.ParamSampling, data.cascadesPositions.Len());
                     sampledCascadesPositions.Add(data.cascadesPositions.GetKey(index));
                  }
                  iterNm++;
               }
               coorIterNm++;
            }

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
      TIntV sampledCascadesPositions;

      void Expectation(UPEMLikelihoodFunction<parameter> &LF, Data data) const {
         for (TIntV::TIter CI = sampledCascadesPositions.BegI(); CI < sampledCascadesPositions.EndI(); CI++) {
            TInt key = data.cascH.GetKey(*CI);
            Datum datum = {data.NodeNmH, data.cascH, key, data.time};

            TInt size = configure.latentVariableSize;
            THash<TInt,TFlt> jointLikelihoodTable;
            for (TInt latentVariable=0; latentVariable < size; latentVariable++) {
               jointLikelihoodTable.AddDat(latentVariable, LF.JointLikelihood(datum,latentVariable));
            }

            //printf("\nindex:%d",CI.GetKey()());
            THash<TInt,TFlt> &latentDistribution = LF.latentDistributions.GetDat(key);
            for (TInt latentVariable=0; latentVariable < size; latentVariable++) {
               TFlt likelihood = 0.0;
               for (TInt i=0; i < size; i++)
                  likelihood += TMath::Power(TMath::E, jointLikelihoodTable.GetDat(i) - jointLikelihoodTable.GetDat(latentVariable));
               latentDistribution.GetDat(latentVariable) = 1.0/likelihood;
               if (latentDistribution.GetDat(latentVariable) < 0.001) latentDistribution.GetDat(latentVariable) = 0.001;
               //printf(", k:%d, p:%f",latentVariable(),latentDistribution.GetDat(latentVariable)());
            }
            //printf("\n\n");
         }
      }
      void Maximization(UPEMLikelihoodFunction<parameter> &LF, Data data) {
         
         coorIterNm = 0;
      
         double time = data.time;
         THash<TInt, TCascade> &cascH = data.cascH;
         TIntFltH &cascadesIdx = data.cascadesPositions, sampledCascadesPositionsHash;
         size_t scale = configure.pGDConfigure.maxIterNm / 1;
         parameter oldParameterDiff;
      
         while(coorIterNm < configure.maxCoorIterNm) {
            //maximize acquaintance
            iterNm = 0;
            size_t sampledIndex = coorIterNm * configure.pGDConfigure.maxIterNm * configure.pGDConfigure.batchSize;
            while(iterNm < configure.pGDConfigure.maxIterNm) { 
               parameter parameterDiff;
               for (size_t i=0;i<configure.pGDConfigure.batchSize;i++) {
                  int index = sampledCascadesPositions[sampledIndex];
                  sampledCascadesPositionsHash.AddDat(cascadesIdx.GetKey(index), 0.0);
                  Datum datum = {data.NodeNmH, cascH, cascH.GetKey(cascadesIdx.GetKey(index)), time};
                  parameterDiff += LF.gradient3(datum);
                  sampledIndex++;
               }
               parameterDiff *= (configure.pGDConfigure.learningRate/double(configure.pGDConfigure.batchSize));
               LF.parameter.projectedlyUpdateGradient(parameterDiff);
               iterNm++;
               if (iterNm % scale == 0) {
                  double size = (double) sampledCascadesPositionsHash.Len();
                  Data sampledData = {data.NodeNmH, data.cascH, sampledCascadesPositionsHash, data.time};
                  loss = LF.PGDFunction<parameter>::loss(sampledData)/size;
                  printf("iterNm: %d, loss: %f\033[0K\r",(int)iterNm,loss());
                  fflush(stdout);
               }
            }
            printf("\n");

            //maximize receiver
            iterNm = 0;
            sampledIndex = coorIterNm * configure.pGDConfigure.maxIterNm * configure.pGDConfigure.batchSize;
            sampledCascadesPositionsHash.Clr();
            while(iterNm < configure.pGDConfigure.maxIterNm) { 
               parameter parameterDiff;
               for (size_t i=0;i<configure.pGDConfigure.batchSize;i++) {
                  int index = sampledCascadesPositions[sampledIndex];
                  sampledCascadesPositionsHash.AddDat(cascadesIdx.GetKey(index), 0.0);
                  Datum datum = {data.NodeNmH, cascH, cascH.GetKey(cascadesIdx.GetKey(index)), time};
                  parameterDiff += LF.gradient1(datum);
                  sampledIndex++;
               }
               parameterDiff *= (configure.pGDConfigure.learningRate/double(configure.pGDConfigure.batchSize));
               if (iterNm > configure.pGDConfigure.maxIterNm * configure.momentumRatio) oldParameterDiff *= configure.finalMomentum;
               else oldParameterDiff *= configure.initialMomentum;
               oldParameterDiff += parameterDiff;
               LF.parameter.projectedlyUpdateGradient(oldParameterDiff);
               iterNm++;
               if (iterNm % scale == 0) {
                  double size = (double) sampledCascadesPositionsHash.Len();
                  Data sampledData = {data.NodeNmH, data.cascH, sampledCascadesPositionsHash, data.time};
                  loss = LF.PGDFunction<parameter>::loss(sampledData)/size;
                  printf("iterNm: %d, loss: %f\033[0K\r",(int)iterNm,loss());
                  fflush(stdout);
               }
            }
            printf("\n");

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
         for (TInt i=0;i<latentVariableSize;i++) datumLoss += latentDistributions.GetDat(datum.index).GetDat(i) * JointLikelihood(datum,i);
         return -1.0 * datumLoss;
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
