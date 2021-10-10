public class Main {

    public static void main(String[] args) {
        NeuralNet xor = new NeuralNet();

        xor.initializeTrainSet();


        // Train for 3000 times(Binary representation)
        int times = 3000;
        int epoch = 0;
        for(int i = 0; i < times; i++) {
            xor.initializeWeights();
            epoch += xor.trainNet();
        }
        epoch /= 100;
        System.out.println("Average epoch: " + epoch);
        xor.saveError();

    }
}
