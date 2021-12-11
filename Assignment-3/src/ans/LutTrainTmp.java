package ans;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class LutTrainTmp {
    /**
     * This is the main program that attempts to train the static robocode LUT data to give a set of hyper-parameters
     * - Prompt user input on neural net parameters (learning rate, momentum, # of hidden nodes)
     * - Create the neural net
     * - For each trial
     * -   Initialize the neural net
     * -   For each epoch, train all samples in training set
     * -     Accumulate total error and write to file
     * -     If total error < threshold then write weights data to file (optional)
     * -     Else repeat training
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        double totalError;
        double acceptError = 0.05;
        int epoch;
        FileWriter errorFile = null, weightFile = null, epochFile = null;

        /**
         * Training data sets (inputs and outputs) for LUT using bipolar representation.
         */
        int maxTrainSet = 960; // Total number of entries in LUT = 4 x 4 x 4 x 3 x 5 = 960
        int numTrainSet = 0;
        double [][] trainInput = new double[maxTrainSet][5];
        double [] trainOutput = new double[maxTrainSet];

        // Neural net training parameters from user input
        NeuralNet.ActFnType actFn = NeuralNet.ActFnType.BIPOLAR;
        double learningRate, momentumTerm;
        int numHidden;
        boolean saveWeight = false; // Y = write trained weights to output file
        int numTrial = 1; // One trial = one complete training cycle to convergence = produce one output file

        // Prompt user input on training parameters
        Scanner userInput = new Scanner(System.in);
        System.out.print("Enter Learning Rate: ");
        learningRate = userInput.nextDouble();

        System.out.print("Enter Momentum: ");
        momentumTerm = userInput.nextDouble();

        System.out.print("Enter number of hidden nodes: ");
        numHidden = userInput.nextInt();

        if (numHidden < 1 || numHidden > 100) {
            System.out.println("Number of hidden nodes must be between 1 and 100");
            System.exit(-1);
        }

        // Set up training data set from LUT saved data file
        try {
            numTrainSet = load(trainInput, trainOutput);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Create and initialize NN
        NeuralNet lutNN = new NeuralNet(actFn,5,numHidden, learningRate, momentumTerm, -1, 1);
        int tmpEpoch = 500000;
        epochFile = lutNN.createFile("epoch_cnt" + learningRate + "_" + tmpEpoch + "_" + momentumTerm + "_" + numHidden + ".txt");

        for (int t = 0; t < numTrial; t++) {
            // Initialize weights and epoch number for each trial
            lutNN.initializeWeights();
            lutNN.zeroWeights();
            epoch = 0;


            // Create output file containing epoch number and total error
            errorFile = lutNN.createFile("lut_out_" +learningRate + "_" + tmpEpoch + "_" + momentumTerm + "_" + numHidden + ".txt");
            lutNN.writeHeader(errorFile);

            // Repeat training by presenting all training data in each epoch in the NN
            // Until total error is less than threshold value.
            double RMSError = 0.0;
            for(int it = 0; it < tmpEpoch; it++) {
                totalError = 0;
                epoch++;

                //for (int i = 0; i < trainInput.length; i++) {
                for (int i = 0; i < numTrainSet; i++) {
                    lutNN.train(trainInput[i], trainOutput[i]);
                    totalError += lutNN.sqError(trainOutput[i], lutNN.activatedOutput);
                }
                RMSError = Math.sqrt(totalError / numTrainSet);

                // Write total error to file after each epoch
                lutNN.writeDetail(errorFile, epoch, RMSError);
            }

            lutNN.closeFile(errorFile);

            // Save weights to weight file if needed
            if (saveWeight) {
                weightFile = lutNN.createFile("wgt_out_" + learningRate + "_" + tmpEpoch + "_" + momentumTerm + "_" + numHidden + ".txt");
                lutNN.save(weightFile);
                lutNN.closeFile(weightFile);
            }

            lutNN.writeDetail(epochFile, t+1, epoch);
        }
        lutNN.closeFile(epochFile);
    }

    /**
     * Load LUT file (from assignment part 2) into training data set
     * @param trainInput array of training input.
     * @param trainOutput array of training output.
     * @return number of training data.
     */
    public static int load(double [][] trainInput, double [] trainOutput) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader("luttest.txt"));
        String line = reader.readLine();
        int z = 0;
        double minQ = Double.MAX_VALUE;
        double maxQ = Double.MIN_VALUE;

        try {
            for (int i = 0; i < trainInput.length; i++) {
                String splitLine[] = line.split("\t");
                int accessCnt = Integer.parseInt(splitLine[2]);
                if (accessCnt > 0) {
                    trainInput[z][0] = Double.parseDouble(splitLine[0].substring(0,1)) + 1;
                    trainInput[z][1] = Double.parseDouble(splitLine[0].substring(1,2)) + 1;
                    trainInput[z][2] = Double.parseDouble(splitLine[0].substring(2,3)) + 1;
                    trainInput[z][3] = Double.parseDouble(splitLine[0].substring(3,4)) + 1;
                    trainInput[z][4] = Double.parseDouble(splitLine[0].substring(4,5)) + 1;
                    trainOutput[z] = Double.parseDouble(splitLine[1]);
                    if (trainOutput[z] < minQ) {
                        minQ = trainOutput[z];
                    }
                    if (trainOutput[z] > maxQ) {
                        maxQ = trainOutput[z];
                    }
                    z++;
                }

                line = reader.readLine();
            }

            // Normalize the Q-values to {-1, 1}
            for (int i = 0; i < z; i++) {
                trainOutput[i] = (trainOutput[i] - minQ) * 2 / (maxQ - minQ) - 1;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            reader.close();
        }

        return z;
    }
}
