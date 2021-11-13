package RLRobot;

import robocode.*;

import java.lang.Math;
import java.util.*;
import java.io.*;

public class LookUpTable implements LUTInterface{

    private int myHP;
    private int enemyHP;
    private int distance;
    private int distanceWall;
    private int actionSize;
    private double[][][][][] LUT;

    public LookUpTable(int myHP, int enemyHP, int distance, int distanceWall, int action) {
        this.myHP = myHP;
        this.enemyHP = enemyHP;
        this.distance = distance;
        this.distanceWall = distanceWall;
        this.actionSize = action;
        LUT = new double[myHP][enemyHP][distance][distanceWall][action];
        this.initialiseLUT();
    }

    @Override
    public void initialiseLUT() {
        for(int i = 0; i < myHP; i++) {
            for(int j = 0; j < enemyHP; j++) {
                for(int k = 0; k < distance; k++) {
                    for(int m = 0; m < distanceWall; m++) {
                        for(int n = 0; n < actionSize; n++) {
                            LUT[i][j][k][m][n] = 0;
                        }
                    }
                }
            }
        }
    }

    public int getRandomAction() {
        Random random = new Random();
        return random.nextInt(actionSize);
    }

    public int getBestAction(int myHP, int enemyHP, int distance, int distanceWall) {
        double maxQ = Double.MIN_VALUE;
        int actionIndex = -1;

        for(int i = 0; i < actionSize; i++) {
            if(LUT[myHP][enemyHP][distance][distanceWall][i] > maxQ) {
                actionIndex = i;
                maxQ = LUT[myHP][enemyHP][distance][distanceWall][i];
            }
        }
        return actionIndex;
    }

    public double getQValue(int myHP, int enemyHP, int distance, int distanceWall, int action) {
        return LUT[myHP][enemyHP][distance][distanceWall][action];
    }

    public void setQValue(int[] x, double argValue) {
        LUT[x[0]][x[1]][x[2]][x[3]][x[4]] = argValue;
    }

    @Override
    public void save(File argFile) {
        PrintStream saveFile = null;

        try {
            saveFile = new PrintStream(new RobocodeFileOutputStream(argFile));
        } catch (IOException e) {
            e.printStackTrace();
        }

        for(int i = 0; i < myHP; i++) {
            for(int j = 0; j < enemyHP; j++) {
                for(int k = 0; k < distance; k++) {
                    for(int m = 0; m < distanceWall; m++) {
                        for(int n = 0; n < actionSize; n++) {
                            String s = String.format("%d,%d,%d,%d,%d,%3f", myHP,enemyHP, distance, distanceWall, actionSize, LUT[myHP][enemyHP][distance][distanceWall][actionSize]);
                            saveFile.println(s);
                        }
                    }
                }
            }
        }
        saveFile.close();
    }

    @Override
    public void load(String argFileName) throws IOException {
        try {
            BufferedReader in = new BufferedReader(new FileReader(argFileName));
            for(int i = 0; i < myHP; i++) {
                for(int j = 0; j < enemyHP; j++) {
                    for(int k = 0; k < distance; k++) {
                        for(int m = 0; m < distanceWall; m++) {
                            for(int n = 0; n < actionSize; n++) {
                                String line = in.readLine();
                                String [] args = line.split(",");
                                System.out.println(line);
                                double q = Double.parseDouble(args[5]);
                                LUT[i][j][k][m][n] = q;
                            }
                        }
                    }
                }
            }
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public double outputFor(double[] X) {
        return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        return 0;
    }
}