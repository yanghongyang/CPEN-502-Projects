package RLRobot;

public class State {

    public enum HP {low, medium, high}; // 自己的血量
    public enum Distance {close, medium, far}; // 和敌人的距离
    public enum DistanceWall {close, medium, far}; // 和墙的距离
    public enum Action {fire, forwardLeft, forwardRight, backwardLeft, backwardRight, forward, backward, left, right}; // 动作：开火，前左，前右，后左，后右
    public enum Operation {scan, performAction}; // 选择扫描还是做动作

}
