package ans;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This interface is common to both the Neural Net and LUT interfaces.
 * The idea is that you should be able to easily switch the LUT
 * for the Neural Net since the interfaces are identical.
 * @date 5 Nov 2021
 * @author Dicky Wong
 *
 */
public interface CommonInterface {
    /**
     * @param x The input vector. An array of doubles.
     * @return The value returned by the LUT or NN for this input vector
     * Note: assume single output value here.  Change to double[] for more generic case
     */
    public double outputFor(double[] x);

    /**
     * This method will tell and train the NN or the LUT the output
     * value that should be mapped to the given input vector. I.e.
     * the desired correct output value for an input.
     * @param x The input vector
     * @param argValue The new value to learn
     */
    public void train(double[] x, double argValue);

    /**
     * A method to write either a LUT or weights of a neural net to a file.
     * @param filename of type String
     */
    public void save(File filename);

    /**
     * Loads the LUT or neural net weights from a file. The load must of course
     * have knowledge of how the data was written out by the save() method.
     * You should raise an error in the case that an attempt is being made to
     * load data into a LUT or neural net whose structure does not match
     * the data in the file (e.g. wrong number of hidden neurons).
     * @param fileName of type File
     */
    public void load(File fileName) throws IOException;
}