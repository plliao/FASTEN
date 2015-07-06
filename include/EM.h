#ifndef EM_H
#define EM_H

#include <Parameter.h>
#include <PGD.h>
#include <cascdynetinf.h>

template <typename parameter>
class EMLikelihoodFunction;

typedef struct {
   PGDConfigure pGDConfigure;
   size_t maxIterNm;
   TInt latentVariableSize;
}EMConfigure;


template<typename parameter>
class EM {
   public:
      void Optimize(EMLikelihoodFunction<parameter> &LF, Data data) {
         EMIterNm = 0;
         while(!IsTerminate()) {

            sampledCascadesPositions.Clr();
            sampledCascadesPositions.Reserve(configure.pGDConfigure.maxIterNm * configure.pGDConfigure.batchSize);
            for (size_t j=0;j<configure.pGDConfigure.maxIterNm;j++) {
               for (size_t i=0;i<configure.pGDConfigure.batchSize;i++) {
                  int index = InfoPathSampler::sample(configure.pGDConfigure.sampling, configure.pGDConfigure.ParamSampling, data.cascadesPositions.Len());
                  sampledCascadesPositions.Add(data.cascadesPositions.GetKey(index));
               }
            }
            Expectation(LF,data);      
            Maximization(LF,data);
            //loss = LF.Loss(data);
            EMIterNm++;
            printf("EM iteration:%d\n",(int)EMIterNm);
            fflush(stdout);
         }
      }
      bool IsTerminate() const {
         return EMIterNm >= configure.maxIterNm; 
      }
      void set(EMConfigure configure) {
         this->configure = configure;;
      }

   private:
      EMConfigure configure;
      size_t iterNm, EMIterNm;
      TFlt loss;
      TIntV sampledCascadesPositions;

      void Expectation(EMLikelihoodFunction<parameter> &LF, Data data) const {
         for (TIntV::TIter CI = sampledCascadesPositions.BegI(); CI < sampledCascadesPositions.EndI(); CI++) {
            TInt key = data.cascH.GetKey(*CI);
            Datum datum = {data.NodeNmH, data.cascH, key, data.time};

            TInt size = configure.latentVariableSize;
            THash<TInt,TFlt> jointLikelihoodTable;
            for (TInt latentVariable=0; latentVariable < size; latentVariable++) {
               jointLikelihoodTable.AddDat(latentVariable, LF.JointLikelihood(datum,latentVariable));
            }

            THash<TInt,TFlt> &latentDistribution = LF.latentDistributions.GetDat(key);
            for (TInt latentVariable=0; latentVariable < size; latentVariable++) {
               TFlt likelihood = 0.0;
               for (TInt i=0; i < size; i++)
                  likelihood += TMath::Power(TMath::E, jointLikelihoodTable.GetDat(i) - jointLikelihoodTable.GetDat(latentVariable));
               latentDistribution.GetDat(latentVariable) = 1.0/likelihood;
               //printf("index:%d, k:%d, p:%f, likelihood:%f\n",CI.GetKey()(),latentVariable(),latentDistribution.GetDat(latentVariable)(),likelihood());
            }
         }
      }
      void Maximization(EMLikelihoodFunction<parameter> &LF, Data data) {
         iterNm = 0;
      
         double time = data.time;
         THash<TInt, TCascade> &cascH = data.cascH;
         size_t scale = configure.pGDConfigure.maxIterNm / 1;
         size_t sampledIndex = 0;
         TIntFltH sampledCascadesPositionsHash;
      
         while(iterNm < configure.pGDConfigure.maxIterNm) { 
            parameter parameterDiff;
            for (size_t i=0;i<configure.pGDConfigure.batchSize;i++, sampledIndex++) {
               int position = sampledCascadesPositions[sampledIndex];
               sampledCascadesPositionsHash.AddDat(position, 0.0);
               Datum datum = {data.NodeNmH, cascH, cascH.GetKey(position), time};
               parameterDiff += LF.gradient(datum);
            }
            parameterDiff *= (configure.pGDConfigure.learningRate/double(configure.pGDConfigure.batchSize));
            LF.parameter.projectedlyUpdateGradient(parameterDiff);
            iterNm++;
            if (iterNm % scale == 0) {
               double size = (double) sampledCascadesPositionsHash.Len();
               Data sampleData = {data.NodeNmH, data.cascH, sampledCascadesPositionsHash, data.time};
               loss = LF.PGDFunction<parameter>::loss(sampleData)/size;
               printf("iterNm: %d, loss: %f\033[0K\r",(int)iterNm,loss());
               fflush(stdout);
            }
         }
         printf("\n");
         LF.maximize(); 
      }
};

template<typename parameter>
class EMLikelihoodFunction : public PGDFunction<parameter> {
   friend class EM<parameter>;
   public:
      virtual TFlt JointLikelihood(Datum datum, TInt latentVariable) const = 0;
      virtual void maximize() = 0;
      TFlt loss(Datum datum) const {
         TFlt datumLoss = 0.0;
         for (TInt i=0;i<latentVariableSize;i++) datumLoss += latentDistributions.GetDat(datum.index).GetDat(i) * JointLikelihood(datum,i);
         return -1.0 * datumLoss;
      }
      void InitLatentVariable(Data data, EMConfigure configure) {
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
