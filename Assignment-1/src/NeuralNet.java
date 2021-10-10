import com.sun.xml.internal.bind.v2.runtime.output.DOMOutput;

public class NeuralNet implements NeuralNetInterface{

    // initialization:
    // 1) 2-input, 4-hidden, 1-output
    // 2) XOR training set
    // 3) weights: [-0.5, 0.5];
    // 4) learning rate: 0.2;
    // 5) momentum: 0.0

    private Integer inputNum = 2;
    private Integer hiddenNum = 4;
    private Integer outputNum = 1;
    private Double learningRate = 0.2;
    private Double momentum = 0.0;
    private Double initWeightCeiling = 0.5;
    private Double initWeightFloor = -0.5;
    private Integer a = 0;
    private Integer b = 1;

    // define layers

    private Double[] inputLayer = new Double[inputNum + 1];
    private Double[] hiddenLayer = new Double[hiddenNum + 1];
    private Double[] outputLayer = new Double[outputNum + 1];

    // weight matrix
    // The first weighing matrix between the input layer and hidden layer
    // The second weighing matrix between the hidden layer and output layer

    private Double[][] w1 = new Double[inputNum + 1][hiddenNum];
    private Double[][] w2 = new Double[hiddenNum + 1][outputNum];

    /* deltaOutput and deltaHidden: the previous weight change */

    private Double[] deltaOutput = new Double[outputNum];
    private Double[] deltaHidden = new Double[hiddenNum];

    // error signal matrix delta

    private Double[][] deltaw1 = new Double[inputNum + 1][hiddenNum + 1];
    private Double[][] deltaw2 = new Double[hiddenNum + 1][outputNum + 1];

    // error

    private Double[] totalError = new Double[outputNum];
    private Double[] singleError = new Double[outputNum];

    // training set

    private Double[][] trainX;
    private Double[][] trainY;

    public NeuralNet() {
    }

    public NeuralNet(Integer inputNum, Integer hiddenNum, Integer outputNum, Double learningRate, Double momentum, Integer a, Integer b) {
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.a = a;
        this.b = b;
    }

    /**
     * Return a bipolar sigmoid of the input X(The activation function)
     * @param x The input
     * @return f(x) = 2 / (1+e(-x)) - 1
     */
    @Override
    public double sigmoid(double x) {
        return 2 / (1 + Math.exp(-x)) - 1;
    }

    @Override
    public double customSigmoid(double x) {
        return (b - a) / (1 + Math.exp(-x)) + a;
    }

    /**
     * Initialization step 1 : initialize the training dataset(XOR dataset)
     */
    public void initializeTrainSet() {
        trainX = new Double[][]{{0d, 0d}, {0d, 1d}, {1d, 0d}, {1d, 1d}};
        trainY = new Double[][]{{0d}, {1d}, {1d}, {0d}};
    }

    /**
     * Initialization step 2 : Initialize the weights to random values.
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialise the last weight change arrays. This is to implement the alpha term.
     */
    @Override
    public void initializeWeights() {
        // initialize w1 matrix
        for(int i = 0; i < inputNum; i++) {
            for(int j = 0; j < hiddenNum; j++) {
                w1[i][j] = Math.random() - 0.5;
            }
        }

        // initialize w2 matrix
        for(int i = 0; i < hiddenNum; i++) {
            for(int j = 0; j < outputNum; j++) {
                w2[i][j] = Math.random() - 0.5;
            }
        }
    }

    /**
     * Initialize the weights to 0.
     */
    @Override
    public void zeroWeights() {
        // w1 & w2 are initialized to zero
    }

    /**
     * Algorithm step 1: Perform a forward propagation
     */
    public void forwardPropagation() {

        // from input layer to hidden layer
        for(int j = 0; j < hiddenNum; j++) {
            // for each node in the hidden layer, compute the Sj
            for(int i = 0; i < inputNum; i++) {
                hiddenLayer[j] += w1[i][j] * inputLayer[i];
            }
        }
        // from hidden layer to output layer
        for(int k = 0; k < outputNum; k++) {
            for(int j = 0; j < hiddenNum; j++) {
                outputLayer[k] += w2[j][k] * hiddenLayer[j];
            }
            outputLayer[k] = sigmoid(outputLayer[k]);
        }
    }

    /**
     * Algorithm step 2: Perform a backward propagation & Update weights
     */
    public void backwardPropagation() {

        // compute the errors of output units
        for(int k = 0; k < outputNum; k++) {
            deltaOutput[k] = 0d;
            deltaOutput[k] = outputLayer[k] * (1 - outputLayer[k]) * singleError[k];
        }

        // update w2
        for(int k = 0; k < outputNum; k++) {
            for(int j = 0; j < hiddenNum; j++) {
                deltaw2[j][k] = momentum * deltaw2[j][k] + learningRate * deltaOutput[k] * hiddenLayer[j];
                w2[j][k] += deltaw2[j][k];
            }
        }

        // compute the errors of hidden units
        for(int j = 0; j < hiddenNum; j++) {
            deltaHidden[j] = 0d;
            for(int k = 0; k < outputNum; k++) {
                deltaHidden[j] += w2[j][k] * deltaOutput[k];
            }
            deltaHidden[j] = deltaHidden[j] * hiddenLayer[j] * (1 - hiddenLayer[j]);
        }

        // update w1
        for(int j = 0; j < hiddenNum; j++) {
            for(int i = 0; i < inputNum; i++) {
                deltaw1[i][j] = momentum * deltaw1[i][j] + learningRate * deltaHidden[j] * inputLayer[i];
                w1[i][j] += deltaw1[i][j];
            }
        }
    }

    /**
     * Calculate the total error
     */
    public Double calTotalError() {
        return 0.0;
    }
}
