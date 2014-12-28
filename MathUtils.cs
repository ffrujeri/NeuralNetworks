using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FollowLine
{
    public class MathUtils
    {
        public static double Sigmoid(double z)
        {
            return 1 / (1 + Math.Exp(-z));
        }

        public static double SigmoidDerivative(double z)
        {
            double s = Sigmoid(z);

            return s * (1 - s);
        }
    }
}
