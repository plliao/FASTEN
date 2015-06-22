#ifndef TIMESHAPINGFUNCTION_H
#define TIMESHAPINGFUNCTION_H

#include <cascdynetinf.h>

class TimeShapingFunction {
   public:
      virtual TFlt Value(TFlt srcTime,TFlt dstTime) const = 0;
      virtual TFlt Integral(TFlt srcTime,TFlt dstTime) const = 0;
      virtual bool Before(TFlt srcTime,TFlt dstTime) const = 0;
};

class EXPShapingFunction : public TimeShapingFunction {
   public:
     TFlt Value(TFlt srcTime,TFlt dstTime) const;
     TFlt Integral(TFlt srcTime,TFlt dstTime) const;
     bool Before(TFlt srcTime,TFlt dstTime) const;
};

class POWShapingFunction : public TimeShapingFunction {
   public:
     POWShapingFunction(TFlt d) : delta(d) {}
     TFlt Value(TFlt srcTime,TFlt dstTime) const;
     TFlt Integral(TFlt srcTime,TFlt dstTime) const;
     bool Before(TFlt srcTime,TFlt dstTime) const;

     TFlt delta;
};

class RAYShapingFunction : public TimeShapingFunction {
   public:
     TFlt Value(TFlt srcTime,TFlt dstTime) const;
     TFlt Integral(TFlt srcTime,TFlt dstTime) const;
     bool Before(TFlt srcTime,TFlt dstTime) const;
};
#endif
