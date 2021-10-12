import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
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
    private double momentum = 0;
    private double initWeightCeiling = 0.5;
    private double initWeightFloor = -0.5;
    private Integer a = -1;
    private Integer b = 1;
    private double errorThreshold = 0.05;

    // record the time of trainning
    private long second = 0;

    // define layers
    // initialized to be zero
    private double[] inputLayer = new double[inputNum + 1];
    private double[] hiddenLayer = new double[hiddenNum + 1];
    private double[] outputLayer = new double[outputNum];

    // weight matrix
    // The first weighing matrix between the input layer and hidden layer
    // The second weighing matrix between the hidden layer and output layer
    // w1 & w2 are initialized to zero
    private double[][] w1 = new double[inputNum + 1][hiddenNum];
    private double[][] w2 = new double[hiddenNum + 1][outputNum];

    /* deltaOutput and deltaHidden: the previous weight change */

    private double[] deltaOutput = new double[outputNum];
    private double[] deltaHidden = new double[hiddenNum];
    // error signal matrix delta

    private double[][] deltaw1 = new double[inputNum + 1][hiddenNum];
    private double[][] deltaw2 = new double[hiddenNum + 1][outputNum];
    // error

    private double[] totalError = new double[outputNum];
    private double[] singleError = new double[outputNum];
    // save the total error in a list

    private List<String> errorList = new LinkedList<>();

    // training set

    private double[][] trainX;
    private double[][] trainY;
    // tell dataset type between binary and bipolar
    private boolean datasetType; // if datasetType is True, it is binary; otherwise it is bipolar

    private int epoch;

    public NeuralNet() {
    }

    public NeuralNet(Integer inputNum, Integer hiddenNum, Integer outputNum, double learningRate, double momentum, Integer a, Integer b, boolean datasetType) {
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.a = a;
        this.b = b;

        this.datasetType = datasetType;
        second = 0;
        epoch = 0;

    }

    /**
     * Return a binary sigmoid of the input X(The activation function)
     * @param x The input
     * @return f(x) = 1 / (1+e(-x))
     */
    @Override
    public double sigmoid(double x) {
        return (double)1 / (1 + Math.exp(-x));
    }
    /**
     * Return a bipolar sigmoid of the input X(The activation function)
     * @param x The input
     * @return f(x) = 2 / (1+e(-x)) - 1
     */
    @Override
    public double customSigmoid(double x) {
        Integer a = -1;
        Integer b = 1;
        return (double)(b - a) / (1 + Math.exp(-x)) + a;
    }

    /**
     * Initialization step 1 : initialize the training dataset(XOR dataset)
     */
    public void initializeTrainSet() {
        if(datasetType) { // binary dataset
            trainX = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            trainY = new double[][]{{0}, {1}, {1}, {0}};
        }
        else { // bipolar dataset
            trainX = new double[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
            trainY = new double[][]{{-1}, {1}, {1}, {-1}};
        }
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
        for(int i = 0; i < inputNum + 1; i++) { // "1" for the bias with no weights
            for(int j = 0; j < hiddenNum; j++) {
                w1[i][j] = Math.random() - 0.5;
            }
        }

        // initialize w2 matrix
        for(int i = 0; i < hiddenNum + 1; i++) { // "1" for the bias with no weights
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
        // assign b0 with 1
        inputLayer[inputNum] = 1;
        // assign b1 with 1
        hiddenLayer[hiddenNum] = 1;
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
            for(int i = 0; i < inputNum + 1; i++) {
                hiddenLayer[j] += w1[i][j] * inputLayer[i];
            }
            // can use custom sigmoid as an activation function
            if(datasetType) // for binary dataset
                {hiddenLayer[j] = sigmoid(hiddenLayer[j]);}
            else // for bipolar dataset
                {hiddenLayer[j] = customSigmoid(hiddenLayer[j]);}
        }
        // from hidden layer to output layer
        for(int k = 0; k < outputNum; k++) {
            for(int j = 0; j < hiddenNum + 1; j++) {
                outputLayer[k] += w2[j][k] * hiddenLayer[j];
            }
            // can use custom sigmoid as an activation function
            if(datasetType) // for binary dataset
                {outputLayer[k] = sigmoid(outputLayer[k]);}
            else{ // for bipolar dataset
                outputLayer[k] = customSigmoid(outputLayer[k]);
            }

        }
    }

    /**
     * Algorithm step 2: Perform a backward propagation & Update weights
     */
    public void backwardPropagation() {

        // compute the errors of output units
        for(int k = 0; k < outputNum; k++) {
            deltaOutput[k] = 0;
            if(datasetType) // for binary dataset
            {deltaOutput[k] = outputLayer[k] * (1 - outputLayer[k]) * singleError[k];}
            else { // for bipolar dataset
                deltaOutput[k] = 0.5 * (1 - Math.pow(outputLayer[k], 2)) * singleError[k];
            }

        }

        // update w2
        for(int k = 0; k < outputNum; k++) {
            for(int j = 0; j < hiddenNum + 1; j++) {
                deltaw2[j][k] = momentum * deltaw2[j][k] + learningRate * deltaOutput[k] * hiddenLayer[j];
                w2[j][k] += deltaw2[j][k];
            }
        }

        // compute the errors of hidden units
        for(int j = 0; j < hiddenNum; j++) {
            deltaHidden[j] = 0;
            for(int k = 0; k < outputNum; k++) {
                deltaHidden[j] += w2[j][k] * deltaOutput[k];
            }
            if(datasetType) // for binary dataset
            {deltaHidden[j] = deltaHidden[j] * hiddenLayer[j] * (1 - hiddenLayer[j]);}
            else { // for bipolar dataset
                deltaHidden[j] = deltaHidden[j] * 0.5 * (1 - Math.pow(hiddenLayer[j], 2));
            }

        }

        // update w1
        for(int j = 0; j < hiddenNum; j++) {
            for(int i = 0; i < inputNum + 1; i++) {
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
        epoch = 0;
        long startTime = System.nanoTime();
        do{
            for(int k = 0; k < outputNum; k++) {
                totalError[k] = 0;
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
            }
            errorList.add(String.valueOf(totalError[0]));
            epoch++;
        } while (totalError[0] > errorThreshold);
        long endTime = System.nanoTime();
        second = endTime - startTime;
        return epoch;
    }
    public void saveError() {
        try {
            FileWriter fileWriter = new FileWriter("./TrainTotalError-"  + datasetType + "-" + epoch + ".txt");
            for(String s : errorList) {
                fileWriter.write(s + "\n");
            }
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveMomentumError() {
        try {
            FileWriter fileWriter = new FileWriter("./TrainMomentumTotalError-"  + datasetType + "-" + epoch + ".txt");
            for(String s : errorList) {
                fileWriter.write(s + "\n");
            }
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void preSaveError() {
        File file = new File("./Error");
        if(!file.isFile()) {
            String[] childFilePath = file.list();
            if(childFilePath != null) {
                for(String path : childFilePath) {
                    File childFile = new File(file.getAbsoluteFile() + "/" + path);
                    childFile.delete();
                }
            }
        }
    }

    public long getSecond() {
        return second;
    }

    public void setSecond(long second) {
        this.second = second;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }
}
