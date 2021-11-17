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
    // visit: track for the used actions
    private int[][][][][] visit;

    public LookUpTable(int myHP, int enemyHP, int distance, int distanceWall, int action) {
        this.myHP = myHP;
        this.enemyHP = enemyHP;
        this.distance = distance;
        this.distanceWall = distanceWall;
        this.actionSize = action;
        LUT = new double[myHP][enemyHP][distance][distanceWall][action];
        visit = new int[myHP][enemyHP][distance][distanceWall][action];
        initialiseLUT();
    }

    @Override
    public void initialiseLUT() {
        for(int i = 0; i < myHP; i++) {
            for(int j = 0; j < enemyHP; j++) {
                for(int k = 0; k < distance; k++) {
                    for(int m = 0; m < distanceWall; m++) {
                        for(int n = 0; n < actionSize; n++) {
                            LUT[i][j][k][m][n] = Math.random();
                            visit[i][j][k][m][n] = 0;
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
        double maxQ = -1;
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
        visit[x[0]][x[1]][x[2]][x[3]][x[4]]++;
    }

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
                            String s = String.format("%d,%d,%d,%d,%d,%3f,%d", myHP,enemyHP, distance, distanceWall, actionSize, LUT[myHP][enemyHP][distance][distanceWall][actionSize],visit[myHP][enemyHP][distance][distanceWall][actionSize]);
                            saveFile.println(s);
                        }
                    }
                }
            }
        }
        saveFile.close();
    }

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
    public int visit(double[] x) throws ArrayIndexOutOfBoundsException {
        if(x.length != 5){
            throw new ArrayIndexOutOfBoundsException();
        }
        else {
            int a = (int)x[0];
            int b = (int)x[1];
            int c = (int)x[2];
            int d = (int)x[3];
            int e = (int)x[4];
            return visit[a][b][c][d][e];
        }
    }

    public double outputFor(double[] X) {

        return 0;
    }

    public double train(double[] X, double argValue) {
        return 0;
    }
}