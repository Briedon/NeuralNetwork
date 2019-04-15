package neural;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public abstract class NeuralNetwork {

    protected ArrayList<Neuron> neuronNetwork;

    public NeuralNetwork(){
    }

    public void load(ArrayList<Neuron> neurons){
        neuronNetwork=neurons;
    }

    public abstract void compute();

}