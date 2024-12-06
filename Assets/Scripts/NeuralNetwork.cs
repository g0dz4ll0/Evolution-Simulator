using UnityEngine;

public class NeuralNetwork : MonoBehaviour
{
    public int[] networkShape = {2, 4, 4, 2};
    public Layer[] layers;

    public void Awake()
    {
        layers = new Layer[networkShape.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(networkShape[i], networkShape[i + 1]);
        }
    }

    public float[] Brain(float[] inputs)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            if (i == 0)
            {
                layers[i].Forward(inputs);
                layers[i].Activation();
            }
            else if (i == layers.Length - 1)
            {
                layers[i].Forward(layers[i - 1].nodeArray);
            }
            else
            {
                layers[i].Forward(layers[i - 1].nodeArray);
                layers[i].Activation();
            }
        }

        return layers[layers.Length - 1].nodeArray;
    }

    public class Layer
    {
        public float[,] weightsArray;
        public float[] biasesArray;
        public float[] nodeArray;

        private int numNodes;
        private int numInputs;

        public Layer(int numInputs, int numNodes)
        {
            this.numNodes = numNodes;
            this.numInputs = numInputs;

            weightsArray = new float[numNodes, numInputs];
            biasesArray = new float[numNodes];
            nodeArray = new float[numNodes];
        }

        public void Forward(float[] inputsArray)
        {
            nodeArray = new float[numNodes];

            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numInputs; j++)
                {
                    nodeArray[i] += weightsArray[i, j] * inputsArray[j];
                }

                nodeArray[i] += biasesArray[i];
            }
        }

        public void Activation()
        {
            for (int i = 0; i < numNodes; i++)
            {
                if (nodeArray[i] < 0)
                {
                    nodeArray[i] = 0;
                }
            }
        }
    }
}
