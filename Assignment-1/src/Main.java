public class Main {

    public static void main(String[] args) {
        boolean binary = true; // using binary
        NeuralNet xor = new NeuralNet(2, 4, 1, 0.2, 0d, 0, 1, binary);

        xor.initializeTrainSet();

        // Train for 100 times(Binary representation)

        int times = 100;
        int epoch = 0;
        xor.preSaveError();
        for(int i = 0; i < times; i++) {
            xor.initializeWeights();
            epoch += xor.trainNet();
            xor.saveError();
        }
        epoch /= 100;

        System.out.println("Average epoch for binary dataset: " + epoch);
        xor.saveError();

        System.out.println("Average epoch: " + epoch);


        binary = false; // using bipolar
        xor = new NeuralNet(2, 4, 1, 0.2, 0d, 0, 1, binary);

        xor.initializeTrainSet();


        // Train for 100 times(Bipolar representation)
        times = 100;
        epoch = 0;
        for(int i = 0; i < times; i++) {
            xor.initializeWeights();
            epoch += xor.trainNet();
        }
        epoch /= 100;
        System.out.println("Average epoch for bipolar dataset: " + epoch);
        xor.saveError();
    }
}
