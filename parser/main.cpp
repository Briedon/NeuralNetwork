#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Sparse>
#include <random>

struct edge{
    int from;
    int to;
    std::string label;
};


struct node{
    int id;
    float value;
    float delta;
    std::vector<float> weights;
    std::string label;
    std::vector<int> input;
    std::vector<int> output;
    int layer;
};

struct graph{
    std::vector<node> nodes;
    std::vector<Eigen::MatrixXf> matrices;
    std::vector<int> from;
    std::vector<int> to;
};

typedef Eigen::Triplet<float> T;
using namespace Eigen;
using namespace std;

int main(int argc, const char * argv[]) {
    ofstream test_text("/mnt/d/parser/edges.txt");
    if(!test_text.is_open()){
        return 1;
    }
    for(int testnumber=1;testnumber<50;testnumber++){
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> dis(0, 1.0);
        uniform_real_distribution<> inpt(0.5, 1.0);
        string line;
        const clock_t begin_time = clock();
        //ifstream graphfile (argv[1]);
        ifstream graphfile ("/mnt/d/parser/generated graphs/genr"+to_string(testnumber*50)+"spar.gml");
        ifstream inputfile ("/mnt/d/parser/generated input/genr"+to_string(testnumber*50)+"spar.txt");
        vector<node> nodes;
        vector<edge> edges;
        vector<int> from;
        vector<int> to;
        nodes.reserve(5000);
        int ed=0;
        if (graphfile.is_open())
        {
            while ( getline(graphfile,line) )
            {
                int l=line.find("node");
                int e=line.find("edge");
                if(l > 0){
                    node new_node=node();
                    getline(graphfile,line);
                    int id=line.find("id");
                    id+=3;
                    string substr=line.substr(id, line.size());
                    new_node.id=stoi(substr);
                    /*getline(graphfile,line);
                    id=line.find("label");
                    id+=6;
                    substr=line.substr(id, line.size());
                    new_node.label=substr;
                    getline(graphfile,line);*/
                    nodes.push_back(new_node);

                }
                if(e >0){
                    edge newEdge=edge();
                    getline(graphfile,line);
                    int i=line.find("source");
                    i+=6;
                    newEdge.from=stoi(line.substr(i,line.size()));
                    from.push_back(stoi(line.substr(i,line.size())));
                    getline(graphfile,line);
                    i=line.find("target");
                    i+=7;
                    newEdge.to=stoi(line.substr(i,line.size()));
                    to.push_back(stoi(line.substr(i,line.size())));
                    /*getline(graphfile,line);
                    i=line.find("label");
                    i+=6;
                    newEdge.label=line.substr(i,line.size());*/
                    //edges.push_back(newEdge);
                    ed++;
                    node nod=nodes[newEdge.to];
                    nod.input.push_back(newEdge.from);
                    nod.weights.push_back(dis(gen));
                    nodes[newEdge.to]=nod;
                    nod=nodes[newEdge.from];
                    nod.output.push_back(newEdge.to);
                    nodes[newEdge.from]=nod;
                }
            }
            graphfile.close();
        }else cout << "Unable to open file nuda\n";
        vector<vector<float>> inputs;
        vector<vector<float>> outputs;
        int number_of_inputs=0;
        if (inputfile.is_open()) {
            while (getline(inputfile, line)) {
                stringstream ss;

                int dif=line.find("] [");
                ss<<line.substr(1,dif-1);
                string temp;
                int num=0;
                char c;
                vector<float> vec;
                vector<float> out;
                inputs.push_back(vec);
                outputs.push_back(out);
                while(!ss.eof()){
                    ss>>temp;
                    num=stoi(temp);
                    inputs[number_of_inputs].push_back(num);
                }

                ss.clear();
                ss<<line.substr(dif+3,line.find("]",dif+1)-4);
                while(!ss.eof()){
                    ss>>temp;
                    num=stoi(temp);
                    outputs[number_of_inputs].push_back(num);
                }
                number_of_inputs++;


            }
            inputfile.close();
        }else cout << "Unable to open file2\n";


        //topological sorting
        vector<T> coeff;
        vector<T> coeffic;
        vector<node> topological;
        int removed=0;
        int layer=0;
        vector<int> to_remove;
        to_remove.reserve(20000);
        for(int i=0; i < nodes.size(); i++){
            node nod=nodes[i];
            removed=0;
            to_remove.clear();
            if(nod.input.size()==0){
                topological.push_back(nod);
                nod.layer=layer;
                nodes[i]=nod;
                for(int i=0;i < from.size();i++) {
                    if (from[i] == nod.id) {
                        from[i]=-1;
                        to[i]=-1;
                    }
                }
            }
        }

        cout<<"test complete phase one\n";
        cout << ed<<"\n";

        bool no_input[nodes.size()];
        vector<vector<int>> layers_input;
        vector<SparseMatrix<float>> layers_matrix;
        vector<SparseMatrix<float>> layers_ones;
        vector<vector<int>> layers_nodes;
        while(topological.size()!=nodes.size()) {
            layer++;
            removed=0;
            for (int i = 0; i < nodes.size(); i++) {
                no_input[i] = true;
            }

            for (node nod:topological) {
                no_input[nod.id] = false;
            }

            for (int edg:to) {
                if(edg!=-1){
                    no_input[edg]=false;
                }
            }
            cout<<"layer "<<layer<<" prepared\n";
            cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC<<"\n";

            for (int i=0; i < nodes.size(); i++) {

                if (no_input[i]== true) {
                    node nod = nodes[i];
                    nod.layer=layer;
                    nodes[i]=nod;
                    topological.push_back(nod);
                    for(int j=0;j < from.size();j++){
                        if (from[j] == nod.id) {
                            from[j]=-1;
                            to[j]=-1;
                        }
                    }
                }
            }
        }

        vector<int> newnumber;
        int old_number[nodes.size()];
        nodes.clear();
        for(int i=0;i<topological.size();i++){
            newnumber.push_back(i);
            old_number[topological[i].id]=i;
        }
        vector<int> inp;
        vector<int> out;
        for(int i:newnumber){
            node nod=topological[i];
            inp.clear();
            for(int j:nod.input){
                inp.push_back(old_number[j]);
            }
            out.clear();
            for (int j:nod.output){
                out.push_back(old_number[j]);
            }
            nod.id=i;
            nod.input.clear();
            nod.input=inp;
            nod.output.clear();
            nod.output=out;
            nodes.push_back(nod);
        }



        for(int number_of_layer=0;number_of_layer<=layer;number_of_layer++) {
            vector<T> push_layer;
            vector<int> input_of_layer;
            vector<node> node_of_layer;
            vector<int> id_layer;
            for (node nod:nodes) {
                if (nod.layer == number_of_layer) {
                    node_of_layer.push_back(nod);
                    id_layer.push_back(nod.id);
                }
            }

            bool found;
            for (node nod:node_of_layer) {
                for (int i:nod.input) {
                    found = true;
                    for (int j:input_of_layer) {
                        if (i == j) {
                            found = false;
                            break;
                        }
                    }
                    if (found) {
                        input_of_layer.push_back(i);
                    }
                }
            }
            sort(input_of_layer.begin(),input_of_layer.end());
            VectorXf bias(node_of_layer.size());
            SparseMatrix<float> matrix(node_of_layer.size(),input_of_layer.size()+1);
            SparseMatrix<float> ones(node_of_layer.size(),input_of_layer.size()+1);
            coeff.clear();
            coeffic.clear();
            int max = 0;
            for (int i = 0; i < node_of_layer.size(); i++) {
                node nod = node_of_layer[i];
                for (int j:nod.input) {
                    int k = 0;
                    for (int l = 0; l < input_of_layer.size(); l++) {
                        if (input_of_layer[l] == j) {
                            break;
                        }else{
                            k++;
                        }
                    }

                    coeff.push_back(T(i,k,dis(gen)));
                    coeffic.push_back(T(i,k,1));
                    //matrix(i,k) = 0.895;
                }
                coeff.push_back(T(i,input_of_layer.size(),1));
                coeffic.push_back(T(i,input_of_layer.size(),1));
                //matrix(i,input_of_layer.size())=0.78248;
            }
            //cout<< coeff<<"\n";
            matrix.setFromTriplets(coeff.begin(),coeff.end());
            ones.setFromTriplets(coeffic.begin(),coeffic.end());
            layers_input.push_back(input_of_layer);
            layers_ones.push_back(ones);
            layers_matrix.push_back(matrix);
            layers_nodes.push_back(id_layer);
        }
        //evaluate
        cout<<"first test\n";
        cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC <<"\n";
        const clock_t begin_time1 = clock();
        VectorXf vec;
        VectorXf res;
        MatrixXf der;
        ArrayXf tes;
        double results[nodes.size()];
        int last_entry=0;
        srand((unsigned int) time(0));
        vector<ArrayXf> layer_results;
        int number_of_layers=layers_matrix.size();
        layer_results.resize(number_of_layers);
        ArrayXf error;
        VectorXf label=VectorXf::Constant(layers_nodes[number_of_layers-1].size(),1,5);
        SparseMatrix<double> derivation;
        VectorXf deltas;
        double learning_rate=0.5;
        double loss;
        //cout<<layers_matrix[number_of_layers-1]<<"\n";
        for(int i=0;i<100;i++) {
            for(int j=0;j<layers_nodes[0].size();j++){
                results[j]=inputs[i%inputs.size()][j];
            }
            for(int j=0;j<layers_nodes[number_of_layers-1].size();j++){
                label[j]=outputs[i%outputs.size()][j];
            }
            //cout<<results[0]<<"\n";
            last_entry=1;
            for (layer = 0; layer < number_of_layers; layer++) {
                vec.resize(layers_input[layer].size()+1);
                for(int j=0;j<layers_input[layer].size();j++){
                    vec(j)=results[layers_input[layer][j]];
                }
                vec(layers_input[layer].size())=1;
                res=layers_matrix[layer] * vec;
                //cout<<layers_matrix[layer]<<"\n";
                //tes = 1/(1+exp(-1*(res.array())));
                tes=log(1+exp(res.array()));
                vec = tes.matrix();
                layer_results[layer]=tes;
                //this is used cause of the jumping neurons
                //we store the results for each neuron into results and create a new vector everytime for the multiplication
                for(int j=0;j<vec.size();j++){
                    results[last_entry]=vec(j);
                    last_entry++;
                }
            }
            error=vec-label;
            //cout<<vec.transpose()<<"\n";

            //add labels output
            loss= error.abs().sum();
            deltas=VectorXf::Zero(nodes.size());
            ofstream test_text("test.txt", ofstream::binary);
            for (layer = number_of_layers-1; layer >=0 ; layer--) {

                vec.resize(layers_input[layer].size() + 1);
                for (int j = 0; j < layers_input[layer].size(); j++) {
                    vec(j) = results[layers_input[layer][j]];
                }
                vec(layers_input[layer].size()) = 1;
                //create a delta error

                tes = MatrixXf::Constant(layer_results[layer].size(), 1, 1).array();

                tes = tes/(tes+exp(-1*layer_results[layer]));
                //change error to deltas
                //tes = layer_results[layer] * (tes - layer_results[layer]);
                tes = tes * error;
                error = tes;

                res = error.matrix();

                res = res.transpose() * layers_matrix[layer];
                error = res.array();
                for (int j = 0; j < layers_input[layer].size(); j++) {
                    deltas[layers_input[layer][j]] = error(j);
                }
                /*res = tes.matrix();
                vec = learning_rate * vec;
                //cout<<tes.transpose()<<"\n";
                der = res * vec.transpose();

                layers_matrix[layer] = layers_matrix[layer] - layers_ones[layer].cwiseProduct(der);*/
                if (layer > 0) {
                    error = VectorXf::Zero(layers_nodes[layer - 1].size());
                    for (int j = 0; j < layers_nodes[layer - 1].size(); j++) {
                        error[j] = deltas[layers_nodes[layer - 1][j]];
                    }
                }
                //ned to learn deltas, vec input,tes the delta
                // the equitation for the backprop is delta * times next layer neuron weighted summed together the derivation of activation function times the input
                //use the adam propagation https://en.wikipedia.org/wiki/Stochastic_gradient_descent
            }
        }
        //cout<<layers_matrix[number_of_layers-1]<<"\n";
        cout<<"finnish approach one\n";
        cout << float( clock () - begin_time1 ) /  CLOCKS_PER_SEC <<"\n";
        test_text << float( clock () - begin_time1 ) /  CLOCKS_PER_SEC <<"\n";
        const clock_t begin_time2 = clock();
        float sum;
        int last;
        /*for(int j=0;j<100;j++) {
            for (node nod:nodes) {
                nod.delta=0;
                if (nod.input.size() == 0) {
                    nod.value = inpt(gen);
                } else {
                    sum = 0;
                    for (int i:nod.input) {
                        sum += nodes[i].value * nod.weights[i];
                    }
                    nod.value = 1/(1+exp(-1*sum));
                }
            }
            last=label.size()-1;
            for (int i=nodes.size()-1;i>=0;i--) {
                node nod=nodes[i];
                if (nod.output.size() == 0) {
                    nod.delta = (label[last]-nod.value)*(nod.value)*(1-nod.value);
                    last--;

                } else {
                    nod.delta*=(nod.value)*(1-nod.value);
                }
                for(int k=0;k<nod.input.size();k++){
                    node nd=nodes[nod.input[k]];
                    nd.delta+=nod.delta*nod.weights[k]*learning_rate;
                    nod.weights[k]=nod.delta*nod.input[k];
                    nodes[nod.input[k]]=nd;
                }
                nodes[i]=nod;
            }
        }
        cout << float( clock () - begin_time2 ) /  CLOCKS_PER_SEC<<"\n";
        test_text << float( clock () - begin_time2 ) /  CLOCKS_PER_SEC << "\n";*/
    }
    test_text.close();
    //nacti input random vec staci jednotka na zacatek, pak sprav metodu ktera vznasobz a piecewise vsechny ostatni veci
    //sprav nieco s tym ze vstupasi nebude usporiadany prerobit vstup podla usporiadnia
    return 0;
}