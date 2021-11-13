package RLRobot;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {

    /**
     * @param X The input vector. An array of doubles.
     * @return The value returned by the LookUpTable or NN for this input vector
     */
    public double outputFor(double [] X);

    /**
     * This method will tell the NN or the LookUpTable the output
     * @param X The input vector
     * @param argValue The new value to learn
     * @return The error in the output for that input vector
     */
    public double train(double[] X, double argValue);

    /**
     * A method to write either a LookUpTable or weights of an neural net to a file.
     * @param argFile of type File
     */
    public void save(File argFile);

    /**
     * Loads the LookUpTable or neural net weights from file. The load must of course
     * have knowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being made
     * to load data into an LookUpTable or neural net whose structure does not match
     * the data in the file.
     * @param argFileName
     * @throws IOException
     */
    public void load(String argFileName) throws IOException;
}
