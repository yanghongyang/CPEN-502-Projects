public class Main {

    public static void main(String[] args) {
        NeuralNet xor = new NeuralNet(2, 4, 1, 0.2, 0d, 0, 1);

        xor.initializeTrainSet();

        // Train for 3000 times(Binary representation)
        int times = 100;
        int epoch = 0;
        xor.preSaveError();
        for(int i = 0; i < times; i++) {
            xor.initializeWeights();
            epoch += xor.trainNet();
            xor.saveError();
        }
        epoch /= 100;
        System.out.println("Average epoch: " + epoch);

    }
}
