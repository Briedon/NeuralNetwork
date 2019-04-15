package neural;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;

@Setter
@Getter
@Builder
public class Neuron {

    private ArrayList<Long> input;
    private ArrayList<Double>[] weights;
    private ArrayList<Double> biases;
    private Integer activationFunction;
    private Integer Id;
    private ArrayList<Double> output;
    private ArrayList<Long> errors;

    @Override
    public String toString() {
        return super.toString();
    }
}
