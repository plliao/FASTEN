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
   double initialMomentum, finalMomentum, momentumRatio, rmsAlpha;
}UPEMConfigure;


template<typename parameter>
class UPEM {
   public:
      void Optimize(UPEMLikelihoodFunction<parameter> &LF, Data data) {
         emIterNm = 0;
 
         while(!IsTerminate()) {

            sampledCascadesPositions.Clr();
            sampledCascadesPositions.Reserve(configure.maxCoorIterNm * configure.pGDConfigure.maxIterNm * configure.pGDConfigure.batchSize);
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
      TFlt loss, oldLoss;
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

            //printf("\nindex:%d", key());
            THash<TInt,TFlt> &latentDistribution = LF.latentDistributions.GetDat(key);
            for (TInt latentVariable=0; latentVariable < size; latentVariable++) {
               TFlt likelihood = 0.0;
               for (TInt i=0; i < size; i++)
                  likelihood += TMath::Power(TMath::E, jointLikelihoodTable.GetDat(i) - jointLikelihoodTable.GetDat(latentVariable));
               latentDistribution.GetDat(latentVariable) = 1.0/likelihood;
               if (latentDistribution.GetDat(latentVariable) < 0.001) latentDistribution.GetDat(latentVariable) = 0.001;
               //if (latentDistribution.GetDat(latentVariable) > 0.95) latentDistribution.GetDat(latentVariable) = 0.95;
               //printf(", k:%d, p:%f",latentVariable(),latentDistribution.GetDat(latentVariable)());
            }
            //printf("\n\n");
         }
      }
      void Maximization(UPEMLikelihoodFunction<parameter> &LF, Data data) {
         
         coorIterNm = 0;
      
         double time = data.time;
         THash<TInt, TCascade> &cascH = data.cascH;
         size_t scale = configure.pGDConfigure.maxIterNm / 1;
         TFltV sigmaes(4); sigmaes.Add(0.0); sigmaes.Add(0.0); sigmaes.Add(0.0); sigmaes.Add(0.0);
      
         while(coorIterNm < configure.maxCoorIterNm) {
            iterNm = 0;
            TIntFltH sampledCascadesPositionsHash;
            size_t sampledIndex = coorIterNm * configure.pGDConfigure.maxIterNm * configure.pGDConfigure.batchSize;
            while(iterNm < configure.pGDConfigure.maxIterNm) { 
               parameter parameterDiff;
               for (size_t i=0;i<configure.pGDConfigure.batchSize;i++, sampledIndex++) {
                  int position = sampledCascadesPositions[sampledIndex];
                  sampledCascadesPositionsHash.AddDat(position, 0.0);
                  Datum datum = {data.NodeNmH, cascH, cascH.GetKey(position), time};
                  parameterDiff += LF.gradient1(datum);
               }
               parameterDiff *= (1.0/double(configure.pGDConfigure.batchSize));
               LF.calculateAverageRMSProp(configure.rmsAlpha, sigmaes, parameterDiff);
               parameterDiff *= configure.pGDConfigure.learningRate;
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
      virtual void calculateRProp(TFlt, parameter&, parameter&) = 0;
      virtual void calculateRMSProp(TFlt, parameter&, parameter&) = 0;
      virtual void calculateAverageRMSProp(TFlt, TFltV&, parameter&) = 0;
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
