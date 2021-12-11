package ans;

import java.awt.*;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

public class MyRobotLUT extends AdvancedRobot {
    /**
     * Reinforcement Learning parameters
     */
    final double alpha = 0.001;   // Learning rate (0 if no learning)
    //final double alpha = 0.0;   // No learning
    final double gamma = 0.9;   // Discount factor
    final double epsilon = 0.1; // Exploration factor (0 if always greedy)

    /**
     * {State, action} definitions
     * - State 1 : myEnergy (3)
     *   - {0-33, 34-67, 68-100}
     * - State 2 : enemyEnergy (3)
     *   - {0-33, 34-67, 68-100}
     * - State 3 : Distance to enemy (4)
     *   - {1-250, 251-500, 501-750, 751-1000}
     * - State 4 : Distance to center (3)
     *   - {0-150, 151-300, 301-500}
     * - Action (5):
     *   - {forward, backward, left, right, fire}
     * Total number of entries in LUT = 3 x 3 x 4 x 3 x 5 = 540
     */
    public enum myEnergy {x1, x2, x3, x4};
    public enum enemyEnergy {y1, y2, y3, x4};
    public enum disToEnemy {d1, d2, d3, d4};
    public enum disToCenter {e1, e2, e3};
    public enum stateAction {a1, a2, a3, a4, a5};
    public enum mode {scan, action};

    public enum policy {on, off};
    policy runPolicy = policy.off;  // Assume off policy

    /**
     * Lookup Table and battle counters - static so that can retain across rounds
     */
    static boolean randomQ = false;

    static public LUT lut = new LUT(
            myEnergy.values().length,
            enemyEnergy.values().length,
            disToEnemy.values().length,
            disToCenter.values().length,
            stateAction.values().length,
            randomQ);

    static int numRounds = 0;
    static int numWins = 0;
    static boolean startBattle = true;

    // Win rate counter: winRate[0] = # of wins in rounds 1-100, winRate[1] = # of wins in rounds 101-200, etc
    static int[] winRate = new int[10000];

    /**
     * Current and previous states (initial value can be any)
     */
    public myEnergy currmyEnergy = myEnergy.x1;
    public enemyEnergy currenemyEnergy = enemyEnergy.y1;
    public disToEnemy currdisToEnemy = disToEnemy.d1;
    public disToCenter currdisToCenter = disToCenter.e1;
    public stateAction currStateAction = stateAction.a1;

    public myEnergy prevmyEnergy;
    public enemyEnergy prevenemyEnergy;
    public disToEnemy prevdisToEnemy;
    public disToCenter prevdisToCenter;
    public stateAction prevStateAction;

    public mode runMode = mode.scan;

    /**
     * Good/bad instant/terminal reward values
     */
    public final double badInstReward = -0.25;
    public final double goodInstReward = 0.5;
    public final double badTermReward = -0.5;
    public final double goodTermReward = 1.0;
    public double currReward = 0.0;

    /**
     * State values (non-quantized) obtained from onScannedRobot()
     */
    double myenergy = 0.0;
    double enenergy = 0.0;
    double dist2enemy = 0.0;
    double dist2center = 0.0;
    double bearing = 0.0;

    @Override
    public void run() {
        /**
         * A battle contains multiple rounds.
         * run() will be called at start of each round.
         * Only load the LUT file at start of battle (instead of start of each round).
         */
        if (startBattle) {
            try {
                lut.load(getDataFile("luttest.txt"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        lut.save(getDataFile("luttest.txt"));
        startBattle = false;    // startBattle is static so that will not load LUT again in next round

        // Color my robot
        setColors(Color.green, Color.gray, Color.orange, Color.black, Color.red);

        while (true) {
            switch (runMode) {
                case scan: {
                    currReward = 0;
                    // Perform enemy scan, control will go to onScannedRobot()
                    turnRadarRight(90);
                    break;
                }
                case action: {
                    if (Math.random() <= epsilon) {
                        currStateAction = exploreAction();
                    }
                    else {
                        currStateAction = greedyAction(myenergy, enenergy, dist2enemy, dist2center);
                    }

                    /**
                     * These are the macro actions performed by the robot.
                     * Performance may differ depending on how the actions are set.
                     */
                    switch (currStateAction) {
                        case a1: {  // forward
                            setAhead(100);
                            execute();
                            break;
                        }
                        case a2: {  // backward
                            setBack(100);
                            execute();
                            break;
                        }
                        case a3: {  // left
                            setTurnLeft(90);
                            setAhead(100);
                            execute();
                            break;
                        }
                        case a4: {  // right
                            setTurnRight(90);
                            setAhead(100);
                            execute();
                            break;
                        }
                        case a5: {  // fire
                            double turn = getHeading() - getGunHeading() + bearing;
                            turnGunRight(normalizeBearing(turn));
                            fire(3);
                            break;
                        }
                        default: {
                            System.out.println("Invalid action = " + currStateAction);
                        }
                    }

                    // Compute Q based on current rewards and update previous Q
                    double[] x = new double[]{
                            prevmyEnergy.ordinal(),
                            prevenemyEnergy.ordinal(),
                            prevdisToEnemy.ordinal(),
                            prevdisToCenter.ordinal(),
                            prevStateAction.ordinal()};

                    lut.train(x, learnQ(currReward));
                    runMode = mode.scan;    // Switch to scan mode
                    break;
                }
                default: {
                    System.out.println("Invalid runMode = " + runMode);
                }
            }
        }
    }

    /**
     * Normalize the bearing of enemy from robot to the range {-180, 180}.
     * @param angle The input bearing in degrees.
     * @return normalized bearing in the range {-180, 180}.
     */
    double normalizeBearing(double angle) {
        while (angle >  180) {
            angle -= 360;
        }
        while (angle < -180) {
            angle += 360;
        }
        return angle;
    }

    /**
     * Return a random action from current state.
     * @return random action.
     */
    public stateAction exploreAction() {
        int x = new Random().nextInt(stateAction.values().length);
        return stateAction.values()[x];
    }

    /**
     * return the greedy action with max Q value.
     * @param myenergy myenergy (actual value).
     * @param enenergy enemy's energy (actual value).
     * @param dist2enemy distance from enemy (actual value).
     * @param dist2center distance from center (actual value).
     * @return action with max Q value.
     */
    public stateAction greedyAction(double myenergy, double enenergy, double dist2enemy, double dist2center) {
        // Quantize state values to LUT indices
        int maxQAction = 0;
        double maxQ = 0.0;
        double[] x = new double[]{quantEnergy(myenergy), quantEnergy(enenergy), quantdist2enemy(dist2enemy), quantdist2center(dist2center), 0};

        // Locate the greedy action giving the maximum Q value
        for (int i = 0; i < stateAction.values().length; i++) {
            x[4] = i;
            if (lut.outputFor(x) >= maxQ) {
                maxQ = lut.outputFor(x);
                maxQAction = i;
            }
        }

        return stateAction.values()[maxQAction];
    }

    /**
     * return the greedy action with max Q value.
     * @param s1 myenemy (enum index).
     * @param s2 enenemy (enum index).
     * @param s3 distance from enemy (enum index).
     * @param s4 distnace from center (enum index).
     * @return action with max Q value.
     */
    public stateAction greedyAction(int s1, int s2, int s3, int s4) {
        // Quantize state values to LUT indices
        int maxQAction = 0;
        double maxQ = 0.0;
        double[] x = new double[]{s1, s2, s3, s4, 0};

        // Locate the greedy action giving the maximum Q value
        for (int i = 0; i < stateAction.values().length; i++) {
            x[4] = i;
            if (lut.outputFor(x) >= maxQ) {
                maxQ = lut.outputFor(x);
                maxQAction = i;
            }
        }

        return stateAction.values()[maxQAction];
    }

    /**
     * return the new Q value based on TD learning.
     * @param reward reward value.
     * @return learned Q value.
     */
    public double learnQ(double reward) {
        stateAction bestAction = greedyAction(
                currmyEnergy.ordinal(),
                currenemyEnergy.ordinal(),
                currdisToEnemy.ordinal(),
                currdisToCenter.ordinal()
        );

        double[] prevSA = new double[]{
                prevmyEnergy.ordinal(),
                prevenemyEnergy.ordinal(),
                prevdisToEnemy.ordinal(),
                prevdisToCenter.ordinal(),
                prevStateAction.ordinal()
        };

        double[] currSA;    // Current state can be either on or off policy

        if (runPolicy == policy.off) {
            currSA = new double[]{
                    currmyEnergy.ordinal(),
                    currenemyEnergy.ordinal(),
                    currdisToEnemy.ordinal(),
                    currdisToCenter.ordinal(),
                    bestAction.ordinal()
            };
        } else {
            currSA = new double[]{
                    currmyEnergy.ordinal(),
                    currenemyEnergy.ordinal(),
                    currdisToEnemy.ordinal(),
                    currdisToCenter.ordinal(),
                    currStateAction.ordinal()
            };
        }

        double prevQ = lut.outputFor(prevSA);
        double currQ = lut.outputFor(currSA);

        return prevQ + alpha * (reward + gamma * currQ - prevQ);
    }

    /**
     * Move away from the wall when hit wall
     */
//    /** have bugs!!!! **/
    public void moveAway() {
        switch (currStateAction) {
            case a1: // if moving forward now
                setBack(50);
                execute();
                break;
            case a2: // if moving backward now
                setAhead(50);
                execute();
                break;
            case a3: // if turning left/right now
            case a4:
            case a5: { // if firing now
                back(20);
                setTurnRight(30);
                setBack(50);
                execute();
                break;
            }
        }
    }

    /**
     * Quantize distance to enemy to state index
     * Distance to enemy : {0..999} -> {0, 1, 2, 3}
     * @param dist2enemy distance from enemy obtained from onScannedRobot() event.
     * @return quantized distance index.
     */
    public int quantdist2enemy(double dist2enemy) {
        final int factor = 250; // quantize factor

        return (int) dist2enemy / factor;
    }

    /**
     * Quantize distance to center to state index
     * @param dist2center distance to center
     * @return quantized bearing index.
     */
    public int quantdist2center(double dist2center) {

        if(dist2center <= 100) {
            return 0;   // very close
        }
        if(dist2center <= 400) {
            return 1;   // near
        }
        return 2;   // far
    }

    /**
     * Quantize self energy to state index
     * Energy : {0..100} -> {0, 1, 2, 3}
     * @param energy energy of my robot obtained from onScannedRobot() event.
     * @return quantized energy index.
     */
    public int quantEnergy(double energy) {
        if (energy == 0) {
            return 0;
        }
        if (energy <= 33) {
            return 1;
        }
        if (energy <= 66) {
            return 2;
        }
        return 3;
    }

    /**
     * Save LUT table to log file
     * @param winArr array of winning count.
     */
    public void saveStats(int[] winArr ) {
        try {
            File winRatesFile = getDataFile("WinRate.txt");
            PrintStream out = new PrintStream(new RobocodeFileOutputStream(winRatesFile));
            out.format("Win rate, %d/%d = %d\n", numWins, numRounds, numWins*100/numRounds);
            out.format("Every 100 Rounds, Wins,\n");
            for (int i = 0; i < (getRoundNum() + 1) / 100; i++) {
                out.format("%d, %d,\n", i + 1, winArr[i]);
            }
            out.close();
        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }

    public double getDistance2Center(double xpos, double ypos, double xCenter, double yCenter) {
        return Math.sqrt(Math.pow((xpos - xCenter), 2) + Math.pow((ypos - yCenter), 2));
    }

    /**
     * Overridden functions OnXXXX for robocode events
     */
    // Update current state based on scanned values
    @Override
    public void onScannedRobot(ScannedRobotEvent e) {
        myenergy = getEnergy();
        enenergy = e.getEnergy();
        dist2enemy = e.getDistance();
        bearing = e.getBearing();
        dist2center = getDistance2Center(getX(), getY(), getBattleFieldWidth() / 2, getBattleFieldHeight() /2);

        // Update previous state
        prevmyEnergy = currmyEnergy;
        prevenemyEnergy = currenemyEnergy;
        prevdisToEnemy = currdisToEnemy;
        prevdisToCenter = currdisToCenter;
        prevStateAction = currStateAction;

        // Update current state
        currmyEnergy = myEnergy.values()[quantEnergy(myenergy)];
        currenemyEnergy = enemyEnergy.values()[quantEnergy(enenergy)];
        currdisToEnemy = disToEnemy.values()[quantdist2enemy(dist2enemy)];
        currdisToCenter = disToCenter.values()[quantdist2center(dist2center)];

        // Switch to action mode
        runMode = mode.action;
    }

    // Hit by enemy robot --> bad instant reward
    @Override
    public void onHitRobot(HitRobotEvent event) {
        currReward = badInstReward;
        moveAway();
    }

    // Enemy hit by bullet --> good instant reward
    @Override
    public void onBulletHit(BulletHitEvent event) {
        currReward = goodInstReward;
    }

    // Hit by enemy bullet --> bad instant reward
    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        currReward = badInstReward;
    }

    // Hit wall --> bad instant reward
    @Override
    public void onHitWall(HitWallEvent e) {
        currReward = badInstReward;
        moveAway();
    }

    @Override
    public void onBulletMissed(BulletMissedEvent e){
        currReward = goodInstReward;
    }

    // Win the round --> good terminal reward
    @Override
    public void onWin(WinEvent event) {
        numWins++;
        currReward = goodTermReward;
        winRate[getRoundNum() / 100]++;

        // Update previous Q before the round ends
        double[] x = new double[]{
                prevmyEnergy.ordinal(),
                prevenemyEnergy.ordinal(),
                prevdisToEnemy.ordinal(),
                prevdisToCenter.ordinal(),
                prevStateAction.ordinal()};

        lut.train(x, learnQ(currReward));
    }

    // Lose the round --> bad terminal reward
    @Override
    public void onDeath(DeathEvent event) {
        currReward = badTermReward;

        // Update previous Q before the round ends
        double[] x = new double[]{
                prevmyEnergy.ordinal(),
                prevenemyEnergy.ordinal(),
                prevdisToEnemy.ordinal(),
                prevdisToCenter.ordinal(),
                prevStateAction.ordinal()};

        lut.train(x, learnQ(currReward));
    }

    // Round ended --> increase number of rounds for winning statistics calculation
    @Override
    public void onRoundEnded(RoundEndedEvent e) {
        numRounds++;
    }

    // Battle ended --> save LUT and battle statistics to file
    @Override
    public void onBattleEnded(BattleEndedEvent e) {
        System.out.println("Win rate = " + numWins + "/" + numRounds);

        // At end of battle, save LUT to file
        lut.save(getDataFile("luttest.txt"));
        saveStats(winRate);
    }
}