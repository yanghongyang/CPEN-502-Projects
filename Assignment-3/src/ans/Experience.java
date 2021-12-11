package ans;

/**
 * Experience stored in replay memory
 * - Previous state
 * - Previous action
 * - Current reward
 * - Current state
 */
public class Experience {
    public State prevState;
    public MyRobotNN.stateAction prevAction;
    public double currReward;
    public State currState;

    // Constructor
    public Experience(State prevState, MyRobotNN.stateAction prevAction, double currReward, State currState) {
        this.prevState = prevState;
        this.prevAction = prevAction;
        this.currReward = currReward;
        this.currState = currState;
    }

    // Convert to string
    @Override
    public String toString() {
        return "[Prev State:" + prevState + "][" +
                "Prev Action:" + prevAction + "][" +
                "Curr Reward:" + currReward + "][" +
                "Curr State:" + currState + "]";
    }
}
