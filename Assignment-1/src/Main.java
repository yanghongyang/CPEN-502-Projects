public class Main {

    public static void main(String[] args) {
        boolean binary = true; // using binary
        NeuralNet xor = new NeuralNet(2, 4, 1, 0.2, 0d, 0, 1, binary);
        NeuralNet xor1 = new NeuralNet(2, 4, 1, 0.2, 0d, 0, 1, binary);
        long second = 0;
        int times = 100;
        int epoch = 0;
        xor.initializeTrainSet();
        xor.preSaveError();
        // Train for 100 times(Binary representation)
        for(int i = 0; i < times; i++) {
            xor.initializeWeights();
            epoch += xor.trainNet();
            second += xor.getSecond();
            xor.saveError();
        }
        epoch /= times;
        second /= times;
        System.out.println("Momentum: " + xor.getMomentum() + ", average epoch for binary dataset: " + epoch + "," + " spend " + second + " nanosecond.");

        // use binary momentum
        xor1.initializeTrainSet();
        xor1.setMomentum(0.9);
        // Train for 100 times(Binary representation)
        second = 0;
        times = 100;
        epoch = 0;
        for(int i = 0; i < times; i++) {
            xor1.initializeWeights();
            epoch += xor1.trainNet();
            second += xor1.getSecond();
            xor1.saveMomentumError();
        }
        epoch /= times;
        second /= times;
        System.out.println("Momentum: " + xor1.getMomentum() + ", average epoch for binary dataset: " + epoch + "," + " spend " + second + " nanosecond.");

        // using bipolar
        binary = false;
        xor = new NeuralNet(2, 4, 1, 0.2, 0d, 0, 1, binary);
        xor.initializeTrainSet();
        // Train for 100 times(Bipolar representation)
        times = 100;
        epoch = 0;
        second = 0;
        for(int i = 0; i < times; i++) {
            xor.initializeWeights();
            epoch += xor.trainNet();
            second += xor.getSecond();
            xor.saveError();
        }
        epoch /= times;
        second /= times;
        System.out.println("Momentum: " + xor.getMomentum() + ", average epoch for bipolar dataset: " + epoch + "," + " spend " + second + " nanoseconds.");

        // use bipolar momentum
        xor1 = new NeuralNet(2, 4, 1, 0.2, 0d, 0, 1, binary);
        xor1.initializeTrainSet();
        xor1.setMomentum(0.9);
        // Train for 100 times(Bipolar representation)
        times = 100;
        epoch = 0;
        second = 0;
        for(int i = 0; i < times; i++) {
            xor1.initializeWeights();
            epoch += xor1.trainNet();
            second += xor1.getSecond();
            xor1.saveMomentumError();
        }
        epoch /= times;
        second /= times;
        System.out.println("Momentum: " + xor1.getMomentum() + ", average epoch for bipolar dataset: " + epoch + "," + " spend " + second + " nanoseconds.");

    }
}
