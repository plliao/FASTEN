#include <UserPropertyModel.h>
#include <kronecker.h>
#include <InfoPathFileIO.h>
#include <cmath>
//#include <iostream>
//using namespace std;

void UserPropertyModel::LoadCascadesTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadCascadesTxt(FIn, CascH, nodeInfo);
}

void UserPropertyModel::LoadGroundTruthTxt(const TStr& InFNm) {
   TFIn FIn(InFNm);
   InfoPathFileIO::LoadNetworkTxt(FIn, Network, nodeInfo);
}

void UserPropertyModel::SaveInferred(const TStr& OutFNm) {
   InfoPathFileIO::SaveNetwork(OutFNm, InferredNetwork, nodeInfo, edgeInfo);
}

void UserPropertyModel::SaveUserProperty(const TStr& OutFNm) {
  TFOut FOut(OutFNm);
  UserPropertyParameter& parameter = lossFunction.getParameter();
  for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
     TInt NId = NI.GetKey();
     FOut.PutStr(TStr::Fmt("%d;", NId()));
     TIntPr index; index.Val1 = NId;
     for (TInt property = 0; property < parameter.configure.propertySize; property++) {
        index.Val2 = property;
        FOut.PutStr(TStr::Fmt("%f", parameter.spreaderProperty.GetDat(index)()));
        if (property != parameter.configure.propertySize-1) FOut.PutStr(",");
     }
     FOut.PutStr(";");

     for (TInt property = 0; property < parameter.configure.propertySize; property++) {
        index.Val2 = property;
        FOut.PutStr(TStr::Fmt("%f", parameter.receiverProperty.GetDat(index)()));
        if (property != parameter.configure.propertySize-1) FOut.PutStr(",");
     }

     FOut.PutStr("\n");
  }
}

void UserPropertyModel::SaveModel(const TStr& OutFNm) {
  TFOut FOut(OutFNm);
  UserPropertyParameter& parameter = lossFunction.getParameter();
  TInt size = parameter.kWeights[0].Len();
  for (THash<TInt, THash<TInt,TFlt> >::TIter WI = parameter.kWeights.BegI(); !WI.IsEnd(); WI++) {
     for (THash<TInt,TFlt>::TIter VI = WI.GetDat().BegI(); !VI.IsEnd(); VI++) {
        FOut.PutStr(TStr::Fmt("%f", VI.GetDat()));
        if (VI.GetKey()!=size-1) FOut.PutStr(",");
     }
     FOut.PutStr("\n");
  }  
}

void UserPropertyModel::ReadUserProperty(const TStr& InFNm) {
  TFIn FIn(InFNm);
  TStr line;
  UserPropertyParameter& parameter = lossFunction.getParameter();
  while (!FIn.Eof()) {
     FIn.GetNextLn(line);
     TStrV tokens, spreaderPropertyStrV, receiverPropertyStrV;
     line.SplitOnAllCh(';', tokens);
     tokens[1].SplitOnAllCh(',', spreaderPropertyStrV);
     tokens[2].SplitOnAllCh(',', receiverPropertyStrV);

     TInt NId = tokens[0].GetInt();
     TInt size = spreaderPropertyStrV.Len();
     TIntPr index(NId, 0);
     for (TInt i=0; i<size; i++) {
        index.Val2 = i;
        parameter.spreaderProperty.AddDat(index, spreaderPropertyStrV[i].GetFlt());
        parameter.receiverProperty.AddDat(index, receiverPropertyStrV[i].GetFlt());
     }
  }
}

void UserPropertyModel::ReadModel(const TStr& InFNm) {
   TFIn FIn(InFNm);
   TStr line;
   UserPropertyParameter& parameter = lossFunction.getParameter();
   TInt topic = 0;
   while (!FIn.Eof()) {
      FIn.GetNextLn(line);
      TStrV tokens;
      line.SplitOnAllCh(',', tokens);

      TInt size = tokens.Len();
      for (TInt i=0; i<size; i++) {
         parameter.kWeights.GetDat(topic).AddDat(i, tokens[i].GetFlt());
      }
      topic++;
   }
}

void UserPropertyModel::GenCascade(TCascade& C) {
	bool verbose = false;
	TIntFltH InfectedNIdH; TIntH InfectedBy;
	double GlobalTime, InitTime;
	double alpha;
	int StartNId;

	if (Network.GetNodes() == 0)
		return;

        // set random seed
        //TInt::Rnd.Randomize();

	while (C.Len() < 2) {
		C.Clr();
		InfectedNIdH.Clr();
		InfectedBy.Clr();

		InitTime = TFlt::Rnd.GetUniDev() * TotalTime; // random starting point <TotalTime
		GlobalTime = InitTime;

		StartNId = Network.GetRndNId();
		InfectedNIdH.AddDat(StartNId) = GlobalTime;

                TInt topic;
                TFlt p = TFlt::Rnd.GetUniDev(), accP = 0.0;
                for (TInt t=0; t < userPropertyFunctionConfigure.parameter.topicSize;t++) {
                   accP += lossFunction.getParameter().kPi.GetDat(t);
                   if (p <= accP) {
                      topic = t;
                      break;
                   } 
                }

		while (true) {
			// sort by time & get the oldest node that did not run infection
			InfectedNIdH.SortByDat(true);
			const int& NId = InfectedNIdH.BegI().GetKey();
			GlobalTime = InfectedNIdH.BegI().GetDat();

			// all the nodes has run infection
			if ( GlobalTime >= TFlt::GetMn(TotalTime, InitTime+Window) )
				break;

			// add current oldest node to the network and set its time
			C.Add(NId, GlobalTime);

			if (verbose) { printf("GlobalTime:%f, infected node:%d\n", GlobalTime, NId); }

			// run infection from the current oldest node
			TStrFltFltHNEDNet::TNodeI NI = Network.GetNI(NId);
			for (int e = 0; e < NI.GetOutDeg(); e++) {
				const int DstNId = NI.GetOutNId(e);

				// choose the current tx rate (we assume the most recent tx rate)
				if (Network.IsEdge(NId,DstNId) && Network.GetEDat(NId, DstNId).Len() > 0) {
				   TFltFltH& Alphas = Network.GetEDat(NId, DstNId);
				   for (int j=0; j<Alphas.Len() && Alphas.GetKey(j)<GlobalTime; j++) { alpha = Alphas[j]; }
                                }
				else alpha = (double)lossFunction.GetAlpha(NId,DstNId,topic);
				if (verbose) { printf("GlobalTime:%f, nodes:%d->%d, alpha:%f\n", GlobalTime, NId, DstNId, alpha); }

				if (alpha<=0.0001) { continue; }

				// not infecting the parent
				if (InfectedBy.IsKey(NId) && InfectedBy.GetDat(NId).Val == DstNId)
					continue;

				double sigmaT;
				switch (nodeInfo.Model) {
				case EXP:
					// exponential with alpha parameter
					sigmaT = TInt::Rnd.GetExpDev(alpha);
					break;
				case POW:
					// power-law with alpha parameter
					sigmaT = TInt::Rnd.GetPowerDev(1+alpha);
					while (sigmaT < Delta) { sigmaT = Delta*TInt::Rnd.GetPowerDev(1+alpha); }
					break;
				case RAY:
					// rayleigh with alpha parameter
					sigmaT = TInt::Rnd.GetRayleigh(1/sqrt(alpha));
					break;
				default:
					sigmaT = 1;
					break;
				}

				IAssert(sigmaT >= 0);

				double t1 = TFlt::GetMn(GlobalTime + sigmaT, TFlt::GetMn(InitTime+Window, TotalTime));

				if (InfectedNIdH.IsKey(DstNId)) {
					double t2 = InfectedNIdH.GetDat(DstNId);
					if ( t2 > t1 && t2 < TFlt::GetMn(InitTime+Window, TotalTime)) {
						InfectedNIdH.GetDat(DstNId) = t1;
						InfectedBy.GetDat(DstNId) = NId;
					}
				} else {
					InfectedNIdH.AddDat(DstNId) = t1;
					InfectedBy.AddDat(DstNId) = NId;
				}
			}

			// we cannot delete key (otherwise, we cannot sort), so we assign a big time (InitTime + window cut-off)
			InfectedNIdH.GetDat(NId) = TFlt::GetMn(InitTime+Window, TotalTime);
		}
    }

	C.Sort();

}

void UserPropertyModel::GenerateGroundTruth(const int& TNetwork, const int& NNodes, const int& NEdges, const TStr& NetworkParams) {
	  bool verbose = true;
	  PNGraph Graph;
	  TKronMtx SeedMtx;
	  TStr MtxNm;

	  // set random seed
	  //TInt::Rnd.Randomize();

	  switch (TNetwork) {
	  // 2-dimension kronecker network
	  case 0:
		  printf("Kronecker graph for Ground Truth\n");
		  SeedMtx = TKronMtx::GetMtx(NetworkParams.CStr()); // 0.5,0.5,0.5,0.5

		  printf("\n*** Seed matrix:\n");
		  SeedMtx.Dump();

		  Graph = TKronMtx::GenFastKronecker(SeedMtx, (int)TMath::Log2(NNodes), NEdges, true, 0);

		  break;

	  // forest fire network
	  case 1:
		  printf("Forest Fire graph for Ground Truth\n");
		  TStrV NetworkParamsV; NetworkParams.SplitOnAllCh(';', NetworkParamsV);

		  TFfGGen FF(true, // BurnExpFireP
					 NetworkParamsV[0].GetInt(), // StartNNodes (1)
					 NetworkParamsV[1].GetFlt(), // ForwBurnProb (0.2)
					 NetworkParamsV[2].GetFlt(), // BackBurnProb (0.17)
					 NetworkParamsV[3].GetInt(), // DecayProb (1)
					 NetworkParamsV[4].GetInt(), // Take2AmbasPrb (0)
					 NetworkParamsV[5].GetInt()); // OrphanPrb (0)

		  FF.GenGraph(NNodes, false);
		  Graph = FF.GetGraph();

		  break;
	  }

	  // fill network structure with graph
	  for (TNGraph::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) { 
             Network.AddNode(NI.GetId()); 
             nodeInfo.NodeNmH.AddDat(NI.GetId(), TNodeInfo(TStr::Fmt("%d", NI.GetId()), 0)); 
          }
	  for (TNGraph::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) { 
             if (EI.GetSrcNId()==EI.GetDstNId()) { continue; } 
             Network.AddEdge(EI.GetSrcNId(),EI.GetDstNId(),TFltFltH()); 
          }

	  if (verbose) { printf("Network structure has been generated succesfully!\n"); }
}

void UserPropertyModel::SaveGroundTruth(TStr fileNm) {
   printf("ground truth\n");
   printf("prior probability:");
   THash<TInt,TFlt>& kPi = lossFunction.getParameter().kPi;
   for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) printf("topic %d: %f, ", piI.GetKey()(), piI.GetDat()());
   printf("\n");

   for (TInt latentVariable=0; latentVariable < userPropertyFunctionConfigure.parameter.topicSize; latentVariable++) {
      TFOut FOut(fileNm + TStr::Fmt("-%d-network.txt", latentVariable+1));
      for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
         FOut.PutStr(TStr::Fmt("%d,%s\n", NI.GetKey().Val, NI.GetDat().Name.CStr()));
      }
      FOut.PutStr("\n");
   }

   for (TStrFltFltHNEDNet::TEdgeI EI = Network.BegEI(); EI < Network.EndEI(); EI++) {
      TInt srcNId = EI.GetSrcNId(), dstNId = EI.GetDstNId();

      printf("%d,%d , \n", srcNId(), dstNId());
      TFlt maxValue = -DBL_MAX;
      TInt topic = -1;         
      for (TInt latentVariable=0; latentVariable < userPropertyFunctionConfigure.parameter.topicSize; latentVariable++) {
         TFlt alpha = lossFunction.GetAlpha(srcNId, dstNId, latentVariable)();
         if (alpha > maxValue) {
            maxValue = alpha;
            topic = latentVariable;
         }
         printf("\t\ttopic %d alpha:%f \n", latentVariable(), alpha());
         if (alpha > edgeInfo.MinAlpha ) {
            TFOut FOut(fileNm + TStr::Fmt("-%d-network.txt", latentVariable+1), true);
            FOut.PutStr(TStr::Fmt("%d,%d,%f,%f\n", srcNId, dstNId, 0.0, alpha));  
         }
      }
      printf("\n");
      EI.GetDat().AddDat(0.0, maxValue);
   }
}

void UserPropertyModel::Init() {
   for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
      InferredNetwork.AddNode(NI.GetKey(), NI.GetDat().Name);
   }
}

extern TFlt sigmoid(TFlt t);

void UserPropertyModel::Infer(const TFltV& Steps, const TStr& OutFNm) {
   
   switch (nodeInfo.Model) {
      case POW :
         userPropertyFunctionConfigure.shapingFunction = new POWShapingFunction(Delta);
         break;
      case RAY :
         userPropertyFunctionConfigure.shapingFunction = new RAYShapingFunction();
         break;
      default :
         userPropertyFunctionConfigure.shapingFunction = new EXPShapingFunction(); 
   } 
   lossFunction.set(userPropertyFunctionConfigure);
   em.set(eMConfigure);
   TIntFltH CascadesPositions;
   Data data = {nodeInfo.NodeNmH, CascH, CascadesPositions, 0.0};
    
   //ExtractFeature();
   lossFunction.InitLatentVariable(data, eMConfigure);
   lossFunction.initParameter(data, userPropertyFunctionConfigure);

   TStr expName, resultDir, outName, modelName;
   OutFNm.SplitOnCh(resultDir, '/', outName);
   outName.SplitOnCh(expName, '-', modelName);
   ReadUserProperty("data/" + expName + "_UserProperty.txt");
   ReadModel("data/" + expName + "_Model.txt");
   //SaveUserProperty(OutFNm + "_InitialUserProperty.txt");
   
   TSampling Sampling = eMConfigure.pGDConfigure.sampling;
   TStrV ParamSamplingV; eMConfigure.pGDConfigure.ParamSampling.SplitOnAllCh(';', ParamSamplingV);

   for (int t=1; t<Steps.Len(); t++) {
      TIntFltH CascadesPositions;
      for (int i=0; i<CascH.Len(); i++) {
         if (CascH[i].LenBeforeT(Steps[t]) > 1 &&
            ( (Sampling!=WIN_SAMPLING && Sampling!=WIN_EXP_SAMPLING) ||
              (Sampling==WIN_SAMPLING && (Steps[t]-CascH[i].GetMinTm()) <= ParamSamplingV[0].GetFlt()) ||
              (Sampling==WIN_EXP_SAMPLING && (Steps[t]-CascH[i].GetMinTm()) <= ParamSamplingV[0].GetFlt()) )) {
            CascadesPositions.AddDat(i) = CascH[i].GetMinTm();
         }
      }
      //Data data = {nodeInfo.NodeNmH, CascH, CascadesPositions, Steps[t]};
      //em.Optimize(lossFunction, data);

      printf("prior probability:");
      const THash<TInt,TFlt>& kPi = lossFunction.getParameter().kPi;
      for (THash<TInt,TFlt>::TIter piI = kPi.BegI(); !piI.IsEnd(); piI++) printf("topic %d: %f, ", piI.GetKey()(), piI.GetDat()());
      printf("\n");

      for (TInt latentVariable = 0; latentVariable < userPropertyFunctionConfigure.parameter.topicSize; latentVariable++) {
         TFOut FOut(OutFNm + TStr("_") + latentVariable.GetStr() + ".txt");
         for (THash<TInt, TNodeInfo>::TIter NI = nodeInfo.NodeNmH.BegI(); NI < nodeInfo.NodeNmH.EndI(); NI++) {
            FOut.PutStr(TStr::Fmt("%d,%s\n", NI.GetKey().Val, NI.GetDat().Name.CStr()));
         }
         FOut.PutStr("\n");
      }

      int i=0;
      for (THash<TInt, TNodeInfo>::TIter SI = nodeInfo.NodeNmH.BegI(); !SI.IsEnd(); SI++) {
         for (THash<TInt, TNodeInfo>::TIter DI = nodeInfo.NodeNmH.BegI(); !DI.IsEnd(); DI++,i++) {
            if (SI.GetKey()== DI.GetKey()) continue;

            TInt srcNId = SI.GetKey(), dstNId = DI.GetKey();
            TIntPr index(srcNId, dstNId);
            if (!lossFunction.allPossibleEdges.IsKey(index)) continue;

            printf("%d,%d:\n", srcNId(), dstNId());
            TFlt maxValue = -DBL_MAX; TInt topic = -1;
            for (TInt latentVariable = 0; latentVariable < userPropertyFunctionConfigure.parameter.topicSize; latentVariable++) {
               TFlt alpha = lossFunction.GetAlpha(srcNId, dstNId, latentVariable);
               if (alpha > maxValue && kPi.GetDat(latentVariable) > 0.0001) { 
                  maxValue = alpha;
                  topic = latentVariable;
               }
               printf("\t\ttopic %d, alpha:%f\n", latentVariable(), alpha());
               if (alpha > edgeInfo.MinAlpha ) {
                  TFOut FOut(OutFNm + TStr("_") + latentVariable.GetStr() + ".txt", true);
                  FOut.PutStr(TStr::Fmt("%d,%d,%f,%f\n", srcNId, dstNId, Steps[t], alpha));  
               }
            }
            printf("\n");

            //if (i%100000==0) printf("add edge: %d,%d , edge size: %d, edge index: %d\n", srcNId(), dstNId(), nodeSize*nodeSize,i);
            TFlt alpha = lossFunction.GetAlpha(srcNId, dstNId, topic);
            /*printf("%d,%d: alpha:%f, acquaintance value:%f, multiplier:%f, property value:%f, topic:%d, topic value:%f\n",\
                   srcNId(), dstNId(), alpha(), acquaintanceValue(), multiplier(), propertyValue(), topic(), topicValue());*/

            if (InferredNetwork.IsEdge(srcNId, dstNId) && InferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t-1]) && 
                alpha == InferredNetwork.GetEDat(srcNId, dstNId).GetDat(Steps[t-1]))
               alpha = alpha * Aging;
            
            if (alpha <= edgeInfo.MinAlpha) continue;
            if (alpha > edgeInfo.MaxAlpha) alpha = edgeInfo.MaxAlpha;
            if (!InferredNetwork.IsEdge(srcNId, dstNId)) InferredNetwork.AddEdge(srcNId, dstNId, TFltFltH());
 
            if (!InferredNetwork.GetEDat(srcNId, dstNId).IsKey(Steps[t])) InferredNetwork.GetEDat(srcNId,dstNId).AddDat(Steps[t]) = alpha;
         }
      }
   }
   delete userPropertyFunctionConfigure.shapingFunction;
}

/*fmat UserPropertyModel::sigmoid(fmat& z) {
   int row = z.n_rows, col = z.n_cols;
   fmat s = zeros<fmat>(row,col);
   for (int i=0;i<row;i++) 
      for (int j=0;j<col;j++)
         s(i,j) = 1.0 / (1.0 + exp(-1.0 * z(i,j)));
   return s;
}

void UserPropertyModel::ExtractFeature() {
   int CascLen = CascH.Len();
   int hiddenSize = userPropertyFunctionConfigure.propertySize;
   int inputSize = nodeInfo.NodeNmH.Len(); 
   int outputSize = nodeInfo.NodeNmH.Len(); 
   int maxEpoch = 100, epoch = 0;

   float learningRate = 1e-3, trainValidRatio = 0.5;
   int trainSize = CascLen * trainValidRatio, validSize = CascLen - trainSize;
   int batchSize = 10;
   int remainSize = trainSize % batchSize;
   int batchTimes = trainSize / batchSize + ((remainSize == 0) ? 0 : 1);
   int validRemainSize = validSize % batchSize;
   int validBatchTimes = validSize / batchSize + ((validRemainSize == 0) ? 0 : 1);
   cout<<"batchSize, "<<batchSize<<" hiddenSize, "<<hiddenSize<<" inputSize, "<<inputSize<<" outputSize, "<<outputSize<<endl;
   cout<<"training remainSize, "<<remainSize<<" training batchTimes, "<<batchTimes<<" trainSize, "<<trainSize<<endl;
   cout<<"valid remainSize, "<<validRemainSize<<" valid batchTimes, "<<validBatchTimes<<" validSize, "<<validSize<<endl;

   fmat W1 = 0.03 * randn<fmat>(hiddenSize, inputSize), W2 = 0.03 * randn<fmat>(outputSize, hiddenSize);
   fvec B1 = 0.03 * randn<fvec>(hiddenSize), B2 = 0.03 * randn<fvec>(outputSize);

   ivec index = shuffle(linspace<ivec>(0, CascLen-1, CascLen));
   ivec trainIndex(trainSize), validIndex(validSize);
   for (int i=0;i<trainSize;i++) trainIndex[i] = index[i];
   for (int i=0;i<validSize;i++) validIndex[i] = index[i + trainSize];

   float validError = 1.0, lastValidError = 1.0;
   while (epoch < maxEpoch && validError <= lastValidError) {
      lastValidError = validError;
      validError = 0.0;
      float trainingError = 0.0;
      ivec randomIndex = shuffle(trainIndex);
      for (int b=0; b < batchTimes; b++) {
         //generate batches
         int currentBatchSize = batchSize;
         if (remainSize != 0 && b == (batchTimes -1)) currentBatchSize = remainSize;
         int nodeNumInCascades = 0;
         for (int i=0; i < currentBatchSize; i++) nodeNumInCascades += CascH[randomIndex[b*batchSize + i]].Len();
         umat locations(2, nodeNumInCascades);
         fvec values(nodeNumInCascades);
         int n = 0;
         for (int i=0; i < currentBatchSize; i++) {
            TCascade& cascade = CascH[randomIndex[b*batchSize + i]];
            for (THash< TInt, THitInfo >::TIter CI = cascade.BegI(); !CI.IsEnd(); CI++) {
               TInt NId = CI.GetDat().NId;
               locations(0,n) = NId();
               locations(1,n) = i;
               values[n] = 1.0;
               n++;
            }
         }
         sp_fmat batch = SpMat<float>(locations, values, inputSize, currentBatchSize);

         //forward
         fmat z1 = W1 * batch + repmat(B1, 1, currentBatchSize);
         fmat a1 = sigmoid(z1);
         fmat z2 = W2 * a1 + repmat(B2, 1, currentBatchSize);
         fmat y = sigmoid(z2);

         //square error
         fmat error = y - batch;

         //back propagate
         float lr = learningRate / float(currentBatchSize);
         fmat delta2 = y % (1.0 - y) % error;
         fmat delta1 = a1 % (1.0 - a1) % (W2.t() * delta2);

         //update
         W2 -= lr * delta2 * a1.t();
         W1 -= lr * delta1 * batch.t();
         B2 -= lr * delta2 * ones<fmat>(currentBatchSize, 1);
         B1 -= lr * delta1 * ones<fmat>(currentBatchSize, 1); 
         trainingError += sum(sum(error % error));
      }

      //evaluate valid error
      for (int b=0; b < validBatchTimes; b++) {
         int currentBatchSize = batchSize;
         if (validRemainSize != 0 && b == (validBatchTimes -1)) currentBatchSize = validRemainSize;
         int nodeNumInValidCascades = 0;
         for (int i=0; i < currentBatchSize; i++) nodeNumInValidCascades += CascH[validIndex[b*batchSize + i]].Len();
         umat locations(2, nodeNumInValidCascades);
         fvec values(nodeNumInValidCascades);
         int n = 0;
         for (int i=0; i < currentBatchSize; i++) {
            TCascade& cascade = CascH[validIndex[b*batchSize + i]];
            for (THash< TInt, THitInfo >::TIter CI = cascade.BegI(); !CI.IsEnd(); CI++) {
               TInt NId = CI.GetDat().NId;
               locations(0,n) = NId();
               locations(1,n) = i;
               values[n] = 1.0;
               n++;
            }
         }
         sp_fmat batch = SpMat<float>(locations, values, inputSize, currentBatchSize);

         fmat z1 = W1 * batch + repmat(B1, 1, currentBatchSize);
         fmat a1 = sigmoid(z1);
         fmat z2 = W2 * a1 + repmat(B2, 1, currentBatchSize);
         fmat y = sigmoid(z2);
         fmat error = y - batch;
         validError += sum(sum(error % error));
      }

      validError /= float(outputSize)*float(validSize);
      cout<<"epoch: "<<epoch+1<<", training error: "<<trainingError/float(outputSize)/float(trainSize)
                              <<", validation error: "<<validError<<"\033[0K\r"<<flush;
      epoch++;
   }
   cout<<endl;

   for (int i=0;i<inputSize;i++) {
      for (int j=0;j<hiddenSize;j++) {
         TIntPr index; index.Val1 = i; index.Val2 = j;
         lossFunction.getParameter().spreaderProperty.AddDat(index, W1(j,i));
         lossFunction.getParameter().receiverProperty.AddDat(index, W2(i,j)); 
      }
   }
}*/ 

