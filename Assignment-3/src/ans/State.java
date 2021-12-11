package ans;

/**
 * State of robot
 * - State 1 : myEnergy (3)
 *   - {0-33, 34-67, 68-100}
 * - State 2 : enemyEnergy (3)
 *   - {0-33, 34-67, 68-100}
 * - State 3 : distToEnemyance to enemy (4)
 *   - {1-250, 251-500, 501-750, 751-1000}
 * - State 4 : distToEnemyance to center (3)
 *   - {0-150, 151-300, 301-500}
 */
public class State {
    public double myEnergy;
    public double enemyEnergy;
    public double distToEnemy;
    public double distToCenter;

    // Constructor
    public State (double myEnergy, double enemyEnergy, double distToEnemy, double distToCenter) {
        this.myEnergy = myEnergy;
        this.enemyEnergy = enemyEnergy;
        this.distToEnemy = distToEnemy;
        this.distToCenter = distToCenter;
    }

    // Get methods
    public double getMyEnergy() {
        return myEnergy;
    }
    public double getEnemyEnergy() { return enemyEnergy; }
    public double getDistToEnemy() { return distToEnemy; }
    public double getDistToCenter() {
        return distToCenter;
    }

    // Set methods
    public void setMyEnergy(double myEnergy) {
        this.myEnergy = myEnergy;
    }
    public void setEnemyEnergy(double enemyEnergy) { this.enemyEnergy = enemyEnergy; }
    public void setDistToEnemy(double distToEnemy) { this.distToEnemy = distToEnemy; }
    public void setDistToCenter(double distToCenter) {
        this.distToCenter = distToCenter;
    }

    // Copy method
    public void copyState(State s) {
        this.myEnergy = s.myEnergy;
        this.enemyEnergy = s.enemyEnergy;
        this.distToEnemy = s.distToEnemy;
        this.distToCenter = s.distToCenter;
    }

    // Convert to string
    @Override
    public String toString() {
        return myEnergy + ", " + enemyEnergy + ", " + distToEnemy + ", " + distToCenter;
    }
}
