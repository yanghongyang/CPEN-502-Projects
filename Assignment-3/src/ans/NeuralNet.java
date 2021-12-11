package ans;

import robocode.RobocodeFileOutputStream;

import java.io.*;

public class NeuralNet implements NeuralNetInterface {
    public enum ActFnType {
        BINARY,
        BIPOLAR,
        CUSTOM
    }

    /**
     * Parameters of the Neural Net
     */
    private ActFnType actFn;
    private int numInputs;  // Dimension of input vector
    private int numHidden;  // Dimension of hidden layer
    // Number of outputs assume = 1
    private double learningRate;    // The learning rate coefficient
    private double momentumTerm;    // The momentum coefficient
    private double asymA;           // sigmoid lower bound
    private double asymB;           // sigmoid upper bound

    /**
     * Data structures of the Neural Net layers
     */
    // Weights of links between neurons will be modelled using m x n array
    // 2-dimensional between input and hidden, 1-dimensional between hidden and output
    double [][] weightsI2H; // Weights from input to hidden layer
    double [] weightsH2O;   // Weights from hidden to output layer (assume single node output)

    // Old weights for computing delta weights
    double [][] oldWeightsI2H;
    double [] oldWeightsH2O;

    // Temp weights for swapping old and new weights
    double [][] tempWeightsI2H;
    double [] tempWeightsH2O;

    // Output signals at neurons (before and after activation) modelled using a 1-dimensional array for hidden layer
    // and single value for output layer
    double [] inducedLocalHidden;
    double [] activatedHidden;
    double inducedLocalOutput = 0;
    public double activatedOutput = 0;

    // Deltas at neurons for back propagation modelled similarly as output signals above
    double [] deltaHidden;
    double deltaOutput = 0;

    public NeuralNet(
            ActFnType actFn,
            int numInputs,
            int numHidden,
            double learningRate,
            double momentumTerm,
            double asymA,
            double asymB) {
        this.actFn = actFn;
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.learningRate = learningRate;
        this.momentumTerm = momentumTerm;
        this.asymA = asymA;
        this.asymB = asymB;

        // Create the neuron layer data structures (+1 is used to include "bias" weight)
        weightsI2H = new double[numHidden][numInputs + 1];
        weightsH2O = new double[numHidden + 1];

        oldWeightsI2H = new double[numHidden][numInputs + 1];
        oldWeightsH2O = new double[numHidden + 1];

        tempWeightsI2H = new double[numHidden][numInputs + 1];
        tempWeightsH2O = new double[numHidden + 1];

        inducedLocalHidden = new double[numHidden];
        activatedHidden = new double[numHidden];

        deltaHidden = new double[numHidden];
    }

    /**
     * Return a binary sigmoid of the input x.
     * @param x The input to sigmoid.
     * @return f(x) = 1 / (1+e(-x)).
     */
    @Override
    public double binarySigmoid(double x) {
        return  1 / (1 + Math.exp(-x));
    }

    /**
     * Return a bipolar sigmoid of the input x.
     * @param x The input to sigmoid.
     * @return f(x) = 2 / (1+e(-x)) - 1.
     */
    @Override
    public double bipolarSigmoid(double x) {
        return  (2 / (1 + Math.exp(-x))) - 1;
    }

    /**
     * Implement a general sigmoid with asymptotes bounded by (a, b).
     * Same as bipolar sigmoid when a=-1, b=1.
     * @param x The input to sigmoid.
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a.
     */
    @Override
    public double customSigmoid(double x) {
        return  ((asymB - asymA) / (1 + Math.exp(-x))) - (-asymA);
    }

    /**
     * Return a derivative of binary sigmoid of the activated input y.
     * @param y The activated input.
     * @return f(y) = y * (1 - y).
     */
    public double deriBinarySigmoid(double y) {
        return  y * (1 - y);
    }

    /**
     * Return a derivative of bipolar sigmoid of the activated input y.
     * @param y The activated input.
     * @return f(y) = 0.5 * (1 + y) * (1 - y).
     */
    public double deriBipolarSigmoid(double y) {
        return  0.5 * (1 + y) * (1 - y);
    }

    /**
     * Initialize the weights to random values between -0.5 and 0.5.
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialize the last weight change arrays. This is to implement the alpha term.
     */
    @Override
    public void initializeWeights() {
        // Randomize input to hidden layer weights
        for (int i = 0; i < weightsI2H.length; i++) {
            for (int j = 0; j < weightsI2H[0].length; j++) {
                weightsI2H[i][j] = Math.random() - 0.5;
            }
        }

        // Randomize hidden to output layer weights
        for (int i = 0; i < weightsH2O.length; i++) {
            weightsH2O[i] = Math.random() - 0.5;
        }
    }

    /**
     * Load a set of weights into NN.  The values can come from a parameter file or hardcoded array.
     * @param loadWeightsI2H The weights in input to hidden layer
     * @param loadWeightsH2O The weights in hidden to output layer
     */
    public void loadWeights(double [][] loadWeightsI2H, double [] loadWeightsH2O) {
        // Load input to hidden layer weights
        for (int i = 0; i < weightsI2H.length; i++) {
            for (int j = 0; j < weightsI2H[0].length; j++) {
                weightsI2H[i][j] = loadWeightsI2H[i][j];
            }
        }

        // Load hidden to output layer weights
        for (int i = 0; i < weightsH2O.length; i++) {
            weightsH2O[i] = loadWeightsH2O[i];
        }
    }

    /**
     * Load a set of weights into NN's old weights.  For JUnit testing purpose.
     * @param loadWeightsI2H The weights in input to hidden layer
     * @param loadWeightsH2O The weights in hidden to output layer
     */
    public void loadOldWeights(double [][] loadWeightsI2H, double [] loadWeightsH2O) {
        // Load input to hidden layer weights
        for (int i = 0; i < oldWeightsI2H.length; i++) {
            for (int j = 0; j < oldWeightsI2H[0].length; j++) {
                oldWeightsI2H[i][j] = loadWeightsI2H[i][j];
            }
        }

        // Load hidden to output layer weights
        for (int i = 0; i < oldWeightsH2O.length; i++) {
            oldWeightsH2O[i] = loadWeightsH2O[i];
        }
    }

    /**
     * Initialize the weights to 0.
     */
    @Override
    public void zeroWeights() {
        // Zero input to hidden layer weights
        for (int i = 0; i < oldWeightsI2H.length; i++) {
            for (int j = 0; j < oldWeightsI2H[0].length; j++) {
                oldWeightsI2H[i][j] = 0;
            }
        }

        // Zero hidden to output layer weights
        for (int i = 0; i < oldWeightsH2O.length; i++) {
            oldWeightsH2O[i] = 0;
        }
    }

    /**
     * Compute output from input vector based on the current model.
     * For NN, this is the forward pass to compute activated signals at both hidden and output layers.
     * @param inputVector The input vector. An array of doubles.
     * @return The value returned by the LUT or NN for this input vector.
     * Note: assume single output value here.  Change to double[] for more generic case.
     */
    @Override
    public double outputFor(double[] inputVector) {
        // Compute weighted sum (induced local) and activated signals at hidden layer
        for (int i = 0; i < numHidden; i++) {
            inducedLocalHidden[i] = 0;

            for (int j = 0; j < numInputs; j++) {
                inducedLocalHidden[i] += inputVector[j] * weightsI2H[i][j];
            }
            inducedLocalHidden[i] += 1.0 * weightsI2H[i][numInputs]; // Add bias weight

            if (actFn == ActFnType.BINARY) {
                activatedHidden[i] = binarySigmoid(inducedLocalHidden[i]);
            } else {
                activatedHidden[i] = bipolarSigmoid(inducedLocalHidden[i]);
            }
        }

        // Compute weighted sum (induced local) and activated signals at output layer
        inducedLocalOutput = 0;
        for (int i = 0; i < numHidden; i++) {
            inducedLocalOutput += activatedHidden[i] * weightsH2O[i];
        }
        inducedLocalOutput += 1.0 * weightsH2O[numHidden]; // Add bias weight

        //System.out.println(inducedLocalOutput);

        if (actFn == ActFnType.BINARY) {
            activatedOutput = binarySigmoid(inducedLocalOutput);
        } else {
            activatedOutput = bipolarSigmoid(inducedLocalOutput);
        }

        return activatedOutput;
    }

    /**
     * Compute the delta (local gradient) at output layer depending on the activation function.
     * @param desiredOutput The expected output value for computing error.
     * Note: deltaOutput contains the result.
     */
    public void bpErrorOutput(double desiredOutput) {
        if (actFn == ActFnType.BINARY) {
            deltaOutput = (desiredOutput - activatedOutput) * deriBinarySigmoid(activatedOutput);
        } else {
            deltaOutput = (desiredOutput - activatedOutput) * deriBipolarSigmoid(activatedOutput);
        }
    }

    /**
     * Compute the delta (local gradient) at hidden layer depending on the activation function.
     * Note: deltaHidden[] contains the result.
     */
    public void bpErrorHidden() {
        for (int i = 0; i < numHidden; i++) {
            if (actFn == ActFnType.BINARY) {
                deltaHidden[i] = weightsH2O[i] * deltaOutput * deriBinarySigmoid(activatedHidden[i]);
            } else {
                deltaHidden[i] = weightsH2O[i] * deltaOutput * deriBipolarSigmoid(activatedHidden[i]);
            }
        }
    }

    /**
     * Update the weights from hidden to output layer using learning rate, momentum and weight delta.
     */
    public void updateWeightsH2O() {
        // Backup current weights
        System.arraycopy(weightsH2O, 0, tempWeightsH2O, 0, weightsH2O.length);

        // Compute delta weight and update weight
        for (int i = 0; i < numHidden; i++) {
            weightsH2O[i] += learningRate * deltaOutput * activatedHidden[i] + momentumTerm * deltaWeightsH2O(i);
        }

        // Update bias weight
        weightsH2O[numHidden] += learningRate * deltaOutput * 1.0 + momentumTerm * deltaWeightsH2O(numHidden);

        // Update old weights
        System.arraycopy(tempWeightsH2O, 0, oldWeightsH2O, 0, weightsH2O.length);
    }

    /**
     * Update the weights from input to hidden layer using learning rate, momentum and weight delta.
     * @param inputVector The input vector from training set.
     */
    public void updateWeightsI2H(double [] inputVector) {
        // Backup current weights
        for (int i = 0; i < numHidden; i++) {
            System.arraycopy(weightsI2H[i], 0, tempWeightsI2H[i], 0, weightsI2H[i].length);
        }

        // Compute delta weight and update weight
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInputs; j++) {
                weightsI2H[i][j] += learningRate * deltaHidden[i] * inputVector[j] + momentumTerm * deltaWeightsI2H(i, j);
            }
            // Update bias weight
            weightsI2H[i][numInputs] += learningRate * deltaHidden[i] * 1.0 + momentumTerm * deltaWeightsI2H(i, numInputs);
        }

        // Update old weights
        for (int i = 0; i < numHidden; i++) {
            System.arraycopy(tempWeightsI2H[i], 0, oldWeightsI2H[i], 0, weightsI2H[i].length);
        }
    }

    /**
     * Updates the delta weight = weight(n) - weight(n-1) where n is the epoch number,
     * for the hidden to output layer.
     * @param i The weight array index.
     * @return The delta weight.
     */
    public double deltaWeightsH2O(int i) {
        if (oldWeightsH2O[i] != 0) {
            return weightsH2O[i] - oldWeightsH2O[i];
        } else {
            return 0;
        }
    }

    /**
     * Update the delta weight = weight(n) - weight(n-1) where n is the epoch number,
     * for the input to hidden layer.
     * @param i, j The weight array indices.
     * @return The delta weight.
     */
    public double deltaWeightsI2H(int i, int j) {
        if (oldWeightsI2H[i][j] != 0) {
            return weightsI2H[i][j] - oldWeightsI2H[i][j];
        } else {
            return 0;
        }
    }

    /**
     * Train the NN or the LUT the output value that should be mapped to the given input vector.
     * I.e. the desired correct output value for an input.
     * @param inputVector The input vector
     * @param desiredOutput The new value to learn
     */
    @Override
    public void train(double[] inputVector, double desiredOutput) {
        outputFor(inputVector);         // Forward pass
        bpErrorOutput(desiredOutput);   // Back propagate output layer error
        updateWeightsH2O();             // Update weights from hidden to output layer
        bpErrorHidden();                // Back propagate hidden layer error
        updateWeightsI2H(inputVector);  // Update weights from input to hidden layer
    }

    /**
     * Return mean squared error based on target and actual values.
     * @param target The target value from training data.
     * @param actual The actual value from NN.
     * @return 0.5 * (target - actual) ^ 2.
     */
    public double meanSqError(double target, double actual) {
        return 0.5 * Math.pow((target - actual), 2);
    }

    /**
     * Return squared error based on target and actual values.
     * @param target The target value from training data.
     * @param actual The actual value from NN.
     * @return (target - actual) ^ 2.
     */
    public double sqError(double target, double actual) {
        return Math.pow((target - actual), 2);
    }

    /**
     * Create the output file for epoch and total error.
     * @param filename The output file name
     * @return FileWriter descriptor
     */
    public FileWriter createFile (String filename) {
        FileWriter file = null;
        try {
            file = new FileWriter(filename);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return file;
    }

    /**
     * Close the output file.
     * @param file The output file
     */
    public void closeFile (FileWriter file) {
        try {
            file.flush();
            file.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Write the header row to file and console.
     * @param file The output file
     */
    public void writeHeader(FileWriter file) {
        System.out.println("Epoch" + "\t" + "Total Error");

        try {
            file.append("Epoch" + "\t" + "Total Error\n");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Write detail line (epoch + total error) to file and console.
     * @param file The output file
     * @param epoch The epoch number
     * @param totalError Total error signal of that epoch
     */
    public void writeDetail(FileWriter file, int epoch, double totalError) {
        System.out.println(epoch + "\t" + totalError);

        try {
            file.append(epoch + "\t" + totalError + "\n");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Save the weights of a neural net to a file to be loaded later when needed (original version).
     * @param file of type File
     */
    public void save(FileWriter file) {
        try {
            file.append("Hidden to Output Layer\n");
            for (int i = 0; i < weightsH2O.length; i++) {
                if (i != weightsH2O.length - 1) {
                    file.append(weightsH2O[i] + "\t");
                } else {
                    file.append(weightsH2O[i] + "\n"); // insert line break for last element in array
                }
            }

            file.append("Input to Hidden Layer\n");
            for (int i = 0; i < weightsI2H.length; i++) {
                for (int j = 0; j < weightsI2H[0].length; j++) {
                    if (j != weightsI2H[0].length - 1) {
                        file.append(weightsI2H[i][j] + "\t");
                    } else {
                        file.append(weightsI2H[i][j] + "\n"); // insert line break for last element in array
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Save the weights of a neural net to a file to be loaded later when needed (robocode version).
     * @param filename of type File
     */
    @Override
    public void save(File filename) {
        PrintStream w = null;
        try {
            w = new PrintStream(new RobocodeFileOutputStream(filename));
            w.println("Hidden to Output Layer");
            for (int i = 0; i < weightsH2O.length; i++) {
                if (i != weightsH2O.length - 1) {
                    w.print(weightsH2O[i] + "\t");
                } else {
                    w.println(weightsH2O[i]); // insert line break for last element in array
                }
            }
            w.println("Input to Hidden Layer");
            for (int i = 0; i < weightsI2H.length; i++) {
                for (int j = 0; j < weightsI2H[0].length; j++) {
                    if (j != weightsI2H[0].length - 1) {
                        w.print(weightsI2H[i][j] + "\t");
                    } else {
                        w.println(weightsI2H[i][j]); // insert line break for last element in array
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            w.flush();
            w.close();
        }
    }

    /**
     * Load the neural net weights from a file. The load must of course
     * have knowledge of how the data was written out by the save() method.
     * An error will be raised in the case that an attempt is being made to
     * load data into a neural net whose structure does not match
     * the data in the file (e.g. wrong number of hidden neurons).
     * @param filename of type File
     * @throws IOException if the input file does not match the neural net structure
     */
    @Override
    public void load(File filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line = reader.readLine(); // Skip comment line 1
        try {
            //int zz = 0;
            while (line != null) {
                //System.out.println(line);
                line = reader.readLine();
                String splitLine[] = line.split("\t");
                for (int i = 0; i < weightsH2O.length; i++) {
                    weightsH2O[i] = Double.valueOf(splitLine[i]);
                }

                line = reader.readLine(); // Skip comment line 2

                for (int i = 0; i < weightsI2H.length; i++) {
                    line = reader.readLine();
                    splitLine = line.split("\t");
                    for (int j = 0; j < weightsI2H[0].length; j++) {
                        weightsI2H[i][j] = Double.valueOf(splitLine[j]);
                    }
                }
                line = reader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            reader.close();
        }
    }
}
