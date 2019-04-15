package neural;

import java.util.ArrayList;
import java.util.Random;

public class StraightForwardApproach extends NeuralNetwork{
    public void compute() {
//        TODO:finnish it
        Random random=new Random();
        for (Neuron neuron:this.neuronNetwork){
            if(neuron.getInput().size()==0){
                neuron.getOutput().add(random.nextDouble());
            }else{
                ArrayList<Double> inputValues=new ArrayList<Double>();
                for(long id:neuron.getInput()){
                    inputValues.add(neuronNetwork.get((int) id).getOutput().get(0));
                }
                double output=0;
                for(int i=0;i<inputValues.size();i++){
                    output+=inputValues.get(i)*neuron.getWeights()[i].get(0);
                }
                neuron.getOutput().add(output);
            }
        }
    }
}
