package ans;

import robocode.*;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

/**
 * Robot using NN to approximate the Q-learning function instead of LUT
 */
public class MyRobotNN extends AdvancedRobot {
    /**
     * Reinforcement Learning parameters
     */
    final double alpha = 0.2;   // Learning rate (0 if no learning)
    //final double alpha = 0.0;   // No learning
    final double gamma = 0.1;   // Discount factor
    final double epsilon = 0.1; // Exploration factor (0 if always greedy)

    /**
     * {State, action} definitions
     * - State 1 : myEnergy (4)
     *   - {0,1-33, 34-67, 68-100}
     * - State 2 : enemyEnergy (4)
     *   - {0,1-33, 34-67, 68-100}
     * - State 3 : dist2enemyance to enemy (4)
     *   - {1-250, 251-500, 501-750, 751-1000}
     * - State 4 : dist2enemyance to center (3)
     *   - {0-150, 151-300, 301-500}
     * - Action (5):
     *   - {forward, backward, left, right, fire}
     * Total number of entries in LUT = 4 x 4 x 4 x 3 x 5 = 960
     */

    public enum stateAction {a1, a2, a3, a4, a5};
    public enum mode {scan, action};
    public enum policy {on, off};
    policy runPolicy = policy.off;  // Assume off policy

    /**
     * Neural net and battle counters - static so that can retain across rounds
     */
    static int numInputs = 5;
    static int numHidden = 24;
    static double learningRate = 0.0001;
    static double momentumTerm = 0.9;
    static public NeuralNet nn = new NeuralNet(
            NeuralNet.ActFnType.BIPOLAR, numInputs, numHidden, learningRate, momentumTerm, -1, 1);

    static int numRounds = 0;
    static int numWins = 0;

    // Win rate counter: winRate[0] = # of wins in rounds 1-100, winRate[1] = # of wins in rounds 101-200, etc
    static int[] winRate = new int[10000];

    /**
     * Create replay memory to train more than 1 sample at a time step
     */
    static int memSize = 10;
    static ReplayMemory<Experience> replayMemory = new ReplayMemory<>(memSize);

    /**
     * Current and previous states (initial value can be any)
     */
    public State currState = new State(100.0, 100.0, 100.0, 20.0);
    public stateAction currStateAction = stateAction.a1;

    public State prevState = new State(100.0, 100.0, 100.0, 20.0);
    public stateAction prevStateAction;

    public mode runMode = mode.scan;

    /**
     * Good/bad instant/terminal reward values
     */
    public final double badInstReward = -0.25;
    public final double goodInstReward = 1.0;
    public final double badTermReward = -0.5;
    public final double goodTermReward = 2.0;
    public double currReward = 0.0;
    public double accumReward = 0.0;

    // rewardRate[0] = accum reward in round 1, rewardRate[1] = accum reward in round 2, etc
    static double[] rewardRate = new double[10000];

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
         * Initialize NN at start of battle (i.e. round 0)
         */

        if (getRoundNum() == 0) {
            nn.initializeWeights();
            nn.zeroWeights();
        }

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
                    // Explore or exploit depending on epsilon
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
                    updatePrevQ();
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
     * Return the greedy action with max Q value.
     * @param myenergy position in x-axis (actual value).
     * @param enenergy position in y-axis (actual value).
     * @param dist2enemy dist2enemyance from enemy (actual value).
     * @param dist2center energy of my robot (actual value).
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
            if (nn.outputFor(x) >= maxQ) {
                maxQ = nn.outputFor(x);
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
    public double learnQ(State prevState, stateAction prevAction, double reward, State currState) {
        stateAction bestAction = greedyAction(
                currState.getMyEnergy(),
                currState.getEnemyEnergy(),
                currState.getDistToEnemy(),
                currState.getDistToCenter()
        );

        double[] prevSA = new double[]{
                prevState.getMyEnergy(),
                prevState.getEnemyEnergy(),
                prevState.getDistToEnemy(),
                prevState.getDistToCenter(),
                prevAction.ordinal()
        };

        double[] currSA;    // Current state can be either on or off policy

        if (runPolicy == policy.off) {
            currSA = new double[]{
                    currState.getMyEnergy(),
                    currState.getEnemyEnergy(),
                    currState.getDistToEnemy(),
                    currState.getDistToCenter(),
                    bestAction.ordinal()
            };
        } else {
            currSA = new double[]{
                    currState.getMyEnergy(),
                    currState.getEnemyEnergy(),
                    currState.getDistToEnemy(),
                    currState.getDistToCenter(),
                    currStateAction.ordinal()
            };
        }

        double prevQ = nn.outputFor(prevSA);
        double currQ = nn.outputFor(currSA);

        return prevQ + alpha * (reward + gamma * currQ - prevQ);
    }

    /**
     * Update Q value of the previous state using learned Q value.
     */
    public void updatePrevQ() {
        double[] x = new double[]{
                prevState.getMyEnergy(),
                prevState.getEnemyEnergy(),
                prevState.getDistToEnemy(),
                prevState.getDistToCenter(),
                prevStateAction.ordinal()};

        replayMemory.add(new Experience(prevState, prevStateAction, currReward, currState));
        replayTrain(x);
    }

    /**
     * Train NN using multiple vectors saved in replayMemory
     */
    public void replayTrain(double[] x) {
        int trainSize = Math.min(replayMemory.sizeOf(), memSize);
        Object[] vector = replayMemory.sample(trainSize);

        for (Object e: vector) {
            Experience exp = (Experience) e;
            nn.train(x, learnQ(exp.prevState,
                    exp.prevAction,
                    exp.currReward,
                    exp.currState));
        }
    }

    /**
     * Move away from the wall when hit wall
     */
//    /** have bugs!!!! **/
    public void moveAway() {
        switch (currStateAction) {
            case a1: // if moving forward now
                setBack(50);
//                setTurnRight(45);
                execute();
                break;
            case a2: // if moving backward now
                setAhead(50);
//                setTurnRight(45);
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
     * Save winning rate table to log file
     * @param winArr array of winning count.
     */
    public void saveStats(int[] winArr) {
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

    /**
     * Save accum reward table to log file
     * @param rewardArr array of winning count.
     */
    public void saveReward(double[] rewardArr ) {
        try {
            File rewardFile = getDataFile("Reward.txt");
            PrintStream out = new PrintStream(new RobocodeFileOutputStream(rewardFile));
            out.format("Round #, Accum reward,\n");
            for (int i = 0; i < getRoundNum() + 1; i++) {
                out.format("%d, %f,\n", i + 1, rewardArr[i]);
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
        prevState.copyState(currState);
        prevStateAction = currStateAction;

        // Update current state
        currState.setMyEnergy(quantEnergy(myenergy));
        currState.setEnemyEnergy(quantEnergy(enenergy));
        currState.setDistToEnemy(quantdist2enemy(dist2enemy));
        currState.setDistToCenter(quantdist2center(dist2center));

        // Switch to action mode
        runMode = mode.action;
    }

    // Hit by enemy robot --> bad instant reward
    @Override
    public void onHitRobot(HitRobotEvent event) {
        currReward += badInstReward;
        accumReward += currReward;
        moveAway();
    }

    // Enemy hit by bullet --> good instant reward
    @Override
    public void onBulletHit(BulletHitEvent event) {
        currReward += goodInstReward;
        accumReward += currReward;
    }

    // Hit by enemy bullet --> bad instant reward
    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        currReward += badInstReward;
        accumReward += currReward;
    }

    // Hit wall --> bad instant reward
    @Override
    public void onHitWall(HitWallEvent e) {
        currReward += badInstReward;
        accumReward += currReward;
        moveAway();
    }

    @Override
    public void onBulletMissed(BulletMissedEvent e){
        currReward += goodInstReward;
    }

    // Win the round --> good terminal reward
    @Override
    public void onWin(WinEvent event) {
        numWins++;
        currReward += goodTermReward;
        accumReward += currReward;
        winRate[getRoundNum() / 100]++;

        // Update previous Q before the round ends
        updatePrevQ();
    }

    // Lose the round --> bad terminal reward
    @Override
    public void onDeath(DeathEvent event) {
        currReward += badTermReward;
        accumReward += currReward;

        // Update previous Q before the round ends
        updatePrevQ();
    }

    // Round ended --> reset reward stats and increase number of rounds for winning statistics calculation
    @Override
    public void onRoundEnded(RoundEndedEvent e) {
        rewardRate[numRounds] = accumReward;
        accumReward = 0; // reset accum reward for next round
        numRounds++;
    }

    // Battle ended --> save NN weights and battle statistics to file
    @Override
    public void onBattleEnded(BattleEndedEvent e) {
        System.out.println("Win rate = " + numWins + "/" + numRounds);

        nn.save(getDataFile("NN_weights.txt")); // Save NN weights
        saveStats(winRate);     // Save winning rate
        saveReward(rewardRate); // Save reward rate
    }
}