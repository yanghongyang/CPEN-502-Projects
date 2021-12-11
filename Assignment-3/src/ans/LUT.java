package ans;

import robocode.*;
import java.io.*;

/**
 * Lookup table for Robocode Reinforcement Learning
 * THis LUT table maps {states, action} to Q value
 * State-action representation (4 states + 1 action):
 * - State 1 : myEnergy (4)
 *   - {0, 1-33, 34-67, 68-100}
 * - State 2 : enemyEnergy (4)
 *   - {0, 1-33, 34-67, 68-100}
 * - State 3 : Distance to enemy (4)
 *   - {1-250, 251-500, 501-750, 751-1000}
 * - State 4 : Distance to center (3)
 *   - {0-150, 151-300, 301-500}
 * - Action (5):
 *   - {forward, backward, left, right, fire}
 * Total number of entries in LUT = 4 x 4 x 4 x 3 x 5 = 540
 */

public class LUT implements CommonInterface {
    private double [][][][][] lut;      // Q value
    private int [][][][][] visited;   // Access count
    private int stateA;              // Dimension for state 1
    private int stateB;              // Dimension for state 2
    private int stateC;              // Dimension for state 3
    private int stateD;              // Dimension for state 4
    private int actionA;              // Dimension for action
    private boolean randomQ;  // Random or zero initial Q

    // Constructor
    public LUT (int stateA, int stateB, int stateC, int stateD, int actionA, boolean randomQ) {
        this.stateA = stateA;
        this.stateB = stateB;
        this.stateC = stateC;
        this.stateD = stateD;
        this.actionA = actionA;
        this.randomQ = randomQ;

        lut = new double [stateA][stateB][stateC][stateD][actionA];
        visited = new int [stateA][stateB][stateC][stateD][actionA];

        this.initLUT();
    }

    /**
     * Initialize the lut array to random number between {0, 1} or zero.
     * Initialize the visited array to 0.
     */
    public void initLUT () {
        for (int a = 0; a < stateA; a++) {
            for (int b = 0; b < stateB; b++) {
                for (int c = 0; c < stateC; c++) {
                    for (int d = 0; d < stateD; d++) {
                        for (int e = 0; e < actionA; e++) {
                            if (randomQ) {
                                lut[a][b][c][d][e] = Math.random();
                            } else {
                                lut[a][b][c][d][e] = 0.0;
                            }
                            visited[a][b][c][d][e] = 0;
                        }

                    }
                }
            }
        }
    }

    /**
     * Return access count of a {state, action} entry.
     * @param x The {state, action} vector.
     * @return access count of the corresponding {state, action} LUT entry.
     */
    public int getvisited (double [] x) {
        return visited[(int)x[0]][(int)x[1]][(int)x[2]][(int)x[3]][(int)x[4]];
    }

    /**
     * Return Q-value of a {state, action} entry, i.e. Q(s, a).
     * @param x The {state, action} vector.
     * @return Q-value of the corresponding {state, action} LUT entry.
     */
    @Override
    public double outputFor (double [] x) {
        return lut[(int)x[0]][(int)x[1]][(int)x[2]][(int)x[3]][(int)x[4]];
    }

    /**
     * Write the current LUT to output file.
     * @param filename Target output file.
    */
    @Override
    public void save(File filename) {
        PrintStream w = null;
        try {
            w = new PrintStream(new RobocodeFileOutputStream(filename));
            for (int a = 0; a < stateA; a++) {
                for (int b = 0; b < stateB; b++) {
                    for (int c = 0; c < stateC; c++) {
                        for (int d = 0; d < stateD; d++) {
                            for (int e = 0; e < actionA; e++) {
                                w.println(a + "" + b + "" + c + "" + d + "" + e + "\t" +
                                        lut[a][b][c][d][e] + "\t" +
                                        visited[a][b][c][d][e]);
                            }
                        }
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
     * Read the saved LUT file into the LUT.
     * @param filename Saved LUT filename.
     */
    @Override
    public void load(File filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line = reader.readLine();
        try {
            int zz = 0;
            while (line != null) {
                String splitLine[] = line.split("\t");
                int a = Character.getNumericValue(splitLine[0].charAt(0));
                int b = Character.getNumericValue(splitLine[0].charAt(1));
                int c = Character.getNumericValue(splitLine[0].charAt(2));
                int d = Character.getNumericValue(splitLine[0].charAt(3));
                int e = Character.getNumericValue(splitLine[0].charAt(4));
                lut[a][b][c][d][e] = Double.valueOf(splitLine[1]);
                visited[a][b][c][d][e] = Integer.valueOf(splitLine[2]);
                line = reader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            reader.close();
        }
    }

    /**
     * Learn the Q-value of {state, action} vector x from argValue.
     * @param x The {state, action} vector.
     * @param target Target value to be learned.
     */
    @Override
    public void train(double[] x, double target) {
        int a = (int)x[0];
        int b = (int)x[1];
        int c = (int)x[2];
        int d = (int)x[3];
        int e = (int)x[4];

        lut[a][b][c][d][e] = target;
        visited[a][b][c][d][e]++;
    }
}