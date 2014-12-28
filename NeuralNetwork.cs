using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FollowLine
{
    public class NeuralNetwork
    {
        private static Random random = new Random();

        private int numInputs;
        private int numHiddens;
        private int numOutputs;

        private double[,] itohWeights;
        private double[,] htooWeights;

        private double alpha;

        public NeuralNetwork(int numInputs, int numHiddens, int numOutputs, double alpha)
        {
            this.numInputs = numInputs;
            this.numHiddens = numHiddens;
            this.numOutputs = numOutputs;
            this.alpha = alpha;

            // +1 to include bias terms
            itohWeights = new double[numInputs + 1, numHiddens];
            htooWeights = new double[numHiddens + 1, numOutputs];

            InitializeWeights(itohWeights);

            InitializeWeights(htooWeights);
        }

        private void InitializeWeights(double[,] weights)
        {
            for (int i = 0; i < weights.GetLength(0); ++i)
            {
                for (int j = 0; j < weights.GetLength(1); ++j)
                {
                    weights[i, j] = 2.0 * random.NextDouble() - 1.0;
                }
            }
        }

        public void Backpropagation(double[] input, double[] expectedOutput)
        {
            double[] inputWithBias = AddBias(input);
            
            double[] hidden = ApplySigmoid(ApplyWeights(inputWithBias, itohWeights));

            double[] hiddenWithBias = AddBias(hidden);

            double[] output = ApplySigmoid(ApplyWeights(hiddenWithBias, htooWeights));

            double[] outputErrors = CalculateErrors(output, expectedOutput);

            double[] hiddenErrors = PropagateErrors(hiddenWithBias, outputErrors, htooWeights);

            UpdateWeights(inputWithBias, hiddenWithBias, hiddenErrors, outputErrors); 
        }

        private void UpdateWeights(double[] input, double[] hidden, double[] hiddenErrors, double[] outputErrors)
        {
            for (int i = 0; i < itohWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < itohWeights.GetLength(1); ++j)
                {
                    itohWeights[i, j] -= alpha * input[i] * hiddenErrors[j];
                }
            }

            for (int i = 0; i < htooWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < htooWeights.GetLength(1); ++j)
                {
                    htooWeights[i, j] -= alpha * hidden[i] * outputErrors[j];
                }
            }
        }

        private double[] PropagateErrors(double[] activation, double[] nextLayerErrors, double[,] weights)
        {
            double[] propagatedErrors = new double[weights.GetLength(0)];

            for (int i = 0; i < weights.GetLength(0); ++i)
            {
                for (int j = 0; j < weights.GetLength(1); ++j)
                {
                    propagatedErrors[i] += weights[i, j] * nextLayerErrors[j] * MathUtils.SigmoidDerivative(activation[i]);
                }
            }

            return propagatedErrors;
        }

        private double[] CalculateErrors(double[] actual, double[] expected)
        {
            double[] errors = new double[actual.Length];

            for (int i = 0; i < actual.Length; ++i)
            {
                errors[i] = actual[i] - expected[i];
            }

            return errors;
        }

        private double[] ApplySigmoid(double[] array)
        {
            for (int i = 0; i < array.Length; ++i)
            {
                array[i] = MathUtils.Sigmoid(array[i]);
            }

            return array;
        }

        private double[] ApplyWeights(double[] array, double[,] weights)
        {
            double[] output = new double[weights.GetLength(1)];

            for (int j = 0; j < weights.GetLength(1); ++j)
            {
                for (int i = 0; i < weights.GetLength(0); ++i)
                {
                    output[j] += weights[i, j] * array[i];
                }
            }

            return output;
        }

        private double[] AddBias(double[] array)
        {
            double[] arrayWithBias = new double[array.Length + 1];

            arrayWithBias[0] = 1.0;

            for (int i = 0; i < array.Length; ++i)
            {
                arrayWithBias[i + 1] = array[i];
            }

            return arrayWithBias;
        }

        public double[] Forwardpropagation(double[] input)
        {
            double[] inputWithBias = AddBias(input);

            double[] hidden = ApplyWeights(inputWithBias, itohWeights);

            double[] hiddenWithBias = AddBias(hidden);

            double[] output = ApplyWeights(hiddenWithBias, htooWeights);

            return output;
        }
    }
}
