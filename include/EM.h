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
         iterNm = 0;
         while(!IsTerminate()) {
            Expectation(LF,data);      
            Maximization(LF,data);
            //loss = LF.Loss(data);
            iterNm++;
            printf("EM iteration:%d\n",(int)iterNm);
            fflush(stdout);
         }
      }
      bool IsTerminate() const {
         return iterNm >= configure.maxIterNm; 
      }
      void set(EMConfigure configure) {
         this->configure = configure;;
         pgd.set(configure.pGDConfigure);
      }

   private:
      EMConfigure configure;
      PGD<parameter> pgd;
      size_t iterNm;
      TFlt loss;

      void Expectation(EMLikelihoodFunction<parameter> &LF, Data data) const {
         for (THash<TInt, TCascade>::TIter CI = data.cascH.BegI(); !CI.IsEnd(); CI++) {
            Datum datum = {data.NodeNmH, data.cascH, CI.GetKey(), data.time};

            TInt size = configure.latentVariableSize;
            THash<TInt,TFlt> jointLikelihoodTable;
            for (TInt latentVariable=0; latentVariable < size; latentVariable++) {
               jointLikelihoodTable.AddDat(latentVariable, LF.JointLikelihood(datum,latentVariable));
            }

            THash<TInt,TFlt> &latentDistribution = LF.latentDistributions.GetDat(CI.GetKey());
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
         pgd.Optimize(LF,data);
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
         for (TInt i=0;i<latentVariableSize;i++) datumLoss += TMath::Power(TMath::E, JointLikelihood(datum,i));
         return datumLoss;
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
