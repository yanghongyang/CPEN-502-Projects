import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class NeuralNet implements NeuralNetInterface{
    // initialization:
    // 1) 2-input, 4-hidden, 1-output
    // 2) XOR training set
    // 3) weights: [-0.5, 0.5];
    // 4) learning rate: 0.2;
    // 5) momentum: 0.0
    // 6) errorThreshold = 0.05;
    private Integer inputNum = 2;
    private Integer hiddenNum = 4;
    private Integer outputNum = 1;
    private double learningRate = 0.2;
    private double momentum = 0.0;
    private double initWeightCeiling = 0.5;
    private double initWeightFloor = -0.5;
    private Integer a = 0;
    private Integer b = 1;
    private double errorThreshold = 0.05;

    // define layers

    private double[] inputLayer = new double[inputNum + 1];
    private double[] hiddenLayer = new double[hiddenNum + 1];
    private double[] outputLayer = new double[outputNum + 1];

    // weight matrix
    // The first weighing matrix between the input layer and hidden layer
    // The second weighing matrix between the hidden layer and output layer

    private double[][] w1 = new double[inputNum + 1][hiddenNum];
    private double[][] w2 = new double[hiddenNum + 1][outputNum];

    /* deltaOutput and deltaHidden: the previous weight change */

    private double[] deltaOutput = new double[outputNum];
    private double[] deltaHidden = new double[hiddenNum];
    // error signal matrix delta
    private double[][] deltaw1 = new double[inputNum + 1][hiddenNum + 1];
    private double[][] deltaw2 = new double[hiddenNum + 1][outputNum + 1];
    // error
    private double[] totalError = new double[outputNum];
    private double[] singleError = new double[outputNum];
    // save the total error in a list
    private List<String> errorList = new LinkedList<>();
    // training set
    private double[][] trainX;
    private double[][] trainY;

    public NeuralNet() {
    }

    public NeuralNet(Integer inputNum, Integer hiddenNum, Integer outputNum, double learningRate, double momentum, Integer a, Integer b) {
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.a = a;
        this.b = b;
        inputLayer = new double[inputNum + 1];
        hiddenLayer = new double[hiddenNum + 1];
        outputLayer = new double[outputNum + 1];
        w1 = new double[inputNum + 1][hiddenNum];
        w2 = new double[hiddenNum + 1][outputNum];
        deltaOutput = new double[outputNum];
        deltaHidden = new double[hiddenNum];
        deltaw1 = new double[inputNum + 1][hiddenNum + 1];
        deltaw2 = new double[hiddenNum + 1][outputNum + 1];
        totalError = new double[outputNum];
        singleError = new double[outputNum];
        errorList = new LinkedList<>();
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
        trainX = new double[][]{{0d, 0d}, {0d, 1d}, {1d, 0d}, {1d, 1d}};
        trainY = new double[][]{{0d}, {1d}, {1d}, {0d}};
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
     * Initialization step 3: Initialize the input layer
     */
    public void initializeInputLayer(double[] sample) {
        for(int i = 0; i < sample.length; i++) {
            inputLayer[i] = sample[i];
        }
    }

    /**
     * Algorithm step 1: Perform a forward propagation
     */
    public void forwardPropagation(double[] sample) {
        // initialize the input layer first
        initializeInputLayer(sample);
        // from input layer to hidden layer
        for(int j = 0; j < hiddenNum; j++) {
            // for each node in the hidden layer, compute the Sj
            for(int i = 0; i < inputNum; i++) {
                hiddenLayer[j] += w1[i][j] * inputLayer[i];
            }
            // can use custom sigmoid as an activation function
            hiddenLayer[j] = sigmoid(hiddenLayer[j]);
        }
        // from hidden layer to output layer
        for(int k = 0; k < outputNum; k++) {
            for(int j = 0; j < hiddenNum; j++) {
                outputLayer[k] += w2[j][k] * hiddenLayer[j];
            }
            // can use custom sigmoid as an activation function
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
     * train the neural network to see the number of epoch
     * */
    public int trainNet() {
        this.errorList.clear();
        int epoch = 0;
        do{
            for(int k = 0; k < outputNum; k++) {
                totalError[k] = 0d;
            }
            // Calculate the total error: 1. calculate the sum and apply the pow
            for(int i = 0; i < trainX.length; i++) {
                double[] sample = trainX[i];
                forwardPropagation(sample);
                for(int k = 0; k < outputNum; k++) {
                    singleError[k] = trainY[i][k] - outputLayer[k];
                    totalError[k] += Math.pow(singleError[k], 2);
                }
                backwardPropagation();
            }
            // Calculate the total error: 2. divide the totalError by 2
            for(int k = 0; k < outputNum; k++) {
                totalError[k] /= 2;
                System.out.println("The total error of output number " + (k + 1) + ": " + totalError[k]);
            }
            errorList.add(epoch + ":" + totalError[0]);
            epoch++;
        } while (totalError[0] > errorThreshold);

        return epoch;
    }
    public void saveError() {
        try {
            Files.write(Paths.get("./trainTotalError.txt"), errorList);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
