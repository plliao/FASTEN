#include <TimeShapingFunction.h>

TFlt EXPShapingFunction::Value(TFlt srcTime,TFlt dstTime) const {
   if (srcTime < dstTime) return 1.0;
   else return 0.0;
}

TFlt EXPShapingFunction::Integral(TFlt srcTime,TFlt dstTime) const {
   if (srcTime < dstTime) return dstTime - srcTime;
   else return 0.0;
}

bool EXPShapingFunction::Before(TFlt srcTime,TFlt dstTime) const {
   return srcTime < dstTime;
}
