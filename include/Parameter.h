#ifndef PARAMETER_H
#define PARAMETER_H

#include <cascdynetinf.h>

struct Datum {
   THash<TInt, TNodeInfo> &NodeNmH;
   THash<TInt, TCascade> &cascH;
   TInt index;
   double time;
};

struct Data {
   THash<TInt, TNodeInfo> &NodeNmH;
   THash<TInt, TCascade> &cascH;
   TIntFltH &cascadesIdx;
   double time;
};

#endif
