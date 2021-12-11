package ans;

public interface NeuralNetInterface extends CommonInterface {
    final double bias = 1.0; //The input for each neuron's bias weight

    // Constructor will be required in the Neural Net interface implementation
    /**
     * Return a bipolar sigmoid of the input x
     * @param x The input
     * @return f(x) = 2 / (1+e(-x)) - 1
     */
    double bipolarSigmoid(double x);

    /**
     * Return a binary sigmoid of the input x
     * @param x The input
     * @return f(x) = 1 / (1+e(-x))
     */
    double binarySigmoid(double x);

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * Same as bipolar sigmoid when a=-1, b=1
     * @param x The input
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
     */
    double customSigmoid(double x);

    /**
     * Initialize the weights to random values.
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialize the last weight change arrays. This is to implement the alpha term.
     */
    void initializeWeights();

    /**
     * Initialize the weights to 0.
     */
    void zeroWeights();

} // End of public interface NeuralNetInterface
