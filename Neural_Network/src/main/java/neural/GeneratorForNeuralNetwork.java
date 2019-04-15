package neural;

import java.sql.Time;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Random;
import java.util.Timer;

public class GeneratorForNeuralNetwork {
    public static final int NUMBER_OF_NEURONS = 40000;
    private static NeuralNetwork neuralNetwork;

    public static void main(String[] args) {
        Long tim=System.currentTimeMillis();
        neuralNetwork = new StraightForwardApproach();
        ArrayList<Neuron> list = generateNeurons(NUMBER_OF_NEURONS);
        list = generateNetwork(list, NUMBER_OF_NEURONS-500, 2, 40);
        list = generateWeights(list);
        neuralNetwork.load(list);
        for (int i = 0; i < 0; i++) {
            neuralNetwork.compute();
        }
        System.out.println((System.currentTimeMillis()-tim));
    }

    private static ArrayList<Neuron> generateNeurons(int numberOfNeurons) {
        ArrayList<Neuron> list = new ArrayList<Neuron>();
        for (int i = 0; i < numberOfNeurons; i++) {
            ArrayList<Double> biases = new ArrayList<Double>();
            ArrayList<Long> input = new ArrayList<Long>();
            ArrayList<Double> output = new ArrayList<Double>();
            ArrayList<Long> errors = new ArrayList<Long>();
            list.add(Neuron.builder()
                    .activationFunction(0)
                    .biases(biases)
                    .Id(i)
                    .errors(errors)
                    .input(input)
                    .output(output)
                    .build());
        }
        return list;
    }

    private static ArrayList<Neuron> generateNetwork(ArrayList<Neuron> list, int firstIdOfLeaves, int smallestNumberofConnections, int biggestNumberOfConnections) {
        Random random = new Random();
        int difference = biggestNumberOfConnections - smallestNumberofConnections;
        for (int i = 0; i < firstIdOfLeaves; i++) {
            int numberOfconnections = random.nextInt(difference) + smallestNumberofConnections;
            int remainingNeurons = firstIdOfLeaves - i;
            for (int j = 0; j < numberOfconnections; j++) {
                int inputNeurons = random.nextInt(remainingNeurons) + i;
                list.get(inputNeurons).getInput().add((long) i);
            }
            list.get(i).getOutput().add((double) 0);
        }
        return list;
    }

    private static ArrayList<Neuron> generateWeights(ArrayList<Neuron> list) {
        Random random = new Random();
        for (Neuron neuron : list) {
            ArrayList<Double>[] weight = new ArrayList[neuron.getInput().size()];
            for (int i = 0; i < neuron.getInput().size(); i++) {
                ArrayList<Double> inputWeight = new ArrayList<Double>();
                for (int j = 0; j < list.get(i).getOutput().size(); j++) {
                    inputWeight.add(random.nextGaussian());
                }
                weight[i] = inputWeight;
            }
            neuron.setWeights(weight);
        }
        return list;
    }
}
