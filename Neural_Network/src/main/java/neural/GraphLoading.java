package neural;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class GraphLoading {

    public static void main(String[] args) {
        File file = new File("C:\\Users\\mbriedon\\Documents\\GitHub\\Neural_Network\\src\\main\\java\\neural\\graphs\\graph0.gml");
        try {
            Scanner scanner = new Scanner(file);

            scanner.hasNextLine();
            scanner.hasNextLine();
            scanner.hasNextLine();
            scanner.hasNextLine();
            scanner.hasNextLine();
            ArrayList<Neuron> neuralNetwork = new ArrayList<Neuron>();
            String scannedLine;
            String id;
            int edge=0;
            int node=0;
            int src,trg;
            while (scanner.hasNextLine()) {
                scannedLine = scanner.nextLine();
                if (scannedLine.contains(" node [")) {
                    scannedLine = scanner.nextLine();
                    id = scannedLine.split("id ")[1];
                    Neuron neuron = Neuron.builder().build();
                    neuron.setId(Integer.valueOf(id));
                    neuron.setInput(new ArrayList<Long>());
                    scannedLine = scanner.nextLine();
                    System.out.println(scannedLine);
                    neuralNetwork.add(neuron);
                    node++;
                }
                if(scannedLine.contains(" edge [")){
                    src=Integer.valueOf(scanner.nextLine().split("source ")[1]);
                    trg=Integer.valueOf(scanner.nextLine().split("target ")[1]);
                    neuralNetwork.get(trg).getInput().add((long)src);
                    edge++;
                }
            }
            System.out.println(edge);
            System.out.println(node);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }
}
