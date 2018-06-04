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
    std::string name;
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
    for(int testnumber=0;testnumber<2;testnumber++){
        random_device rd;
        mt19937 gen(rd());
        //vytvorenie a delkarovanie
        uniform_real_distribution<> dis(0, 1.0);
        uniform_real_distribution<> inpt(0.5, 1.0);
        string line;
        //ifstream graphfile (argv[1]);
        ifstream graphfile ("../graphs/graph"+to_string(testnumber)+".gml");
        //vstupy su generovane nahodne
        int generated_input=0;
        ifstream inputfile;
        if(generated_input==1){
            ifstream inputfile ("../generated input/genr"+to_string(testnumber*50)+"spar.txt");
        }
       //deklarovanie poli vrcholov a hran
        vector<node> nodes;
        vector<int> from;
        vector<int> to;
        //predvyplnenie vrcholov
        nodes.reserve(5000);
        int ed=0;
        if (graphfile.is_open())
        {
            //citanie gml suboru po riadkoch a ukladanie vrcholov
            while ( getline(graphfile,line) )
            {
                //vyskusanie ci dany riadok nie je hlavicka pre
                int l=line.find("node");
                int e=line.find("edge");
                if(l > 0){
                    //novy vrchol
                    node new_node=node();
                    getline(graphfile,line);
                    int id=line.find("id");
                    id+=3;
                    string substr=line.substr(id, line.size());
                    //vybratie a pridanie id do noveho vrcholu
                    new_node.id=stoi(substr);
                    getline(graphfile,line);
                    id = line.find("label");
                    //ak tam je meno tak sa prida
                    if(id>1) {
                        id += 6;
                        substr = line.substr(id, line.size());
                        new_node.label = substr;
                        int name_part=substr.find("\n");
                        new_node.name=substr.substr(0,name_part);
                    }
                    nodes.push_back(new_node);
                }
                if(e >0){
                    //pridanie hrany
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
                    getline(graphfile,line);
                    i=line.find("label");
                    if(i>0) {
                        i += 6;
                        newEdge.label = line.substr(i, line.size());
                    }
                    //pridanie hrany do grafu
                    ed++;
                    node nod=nodes[newEdge.to];
                    //pridanie vstupu a vystupu do prislusnych vrcholov
                    nod.input.push_back(newEdge.from);
                    nod.weights.push_back(dis(gen));
                    nodes[newEdge.to]=nod;
                    nod=nodes[newEdge.from];
                    nod.output.push_back(newEdge.to);
                    nodes[newEdge.from]=nod;
                }
            }
            graphfile.close();
        }else {
            cout << "Unable to open file\n";
            continue;
        }
        //koniec nacitavania dat zo suboru

        vector<vector<float>> inputs;
        vector<vector<float>> outputs;
        int number_of_inputs=0;
        //sluzi na nacitanie vstupov vygenerovanych automaticky
        if(generated_input==1) {
            if (inputfile.is_open()) {
                while (getline(inputfile, line)) {
                    stringstream ss;
                    //rozdeli riadok na vstup a vystup
                    int dif = line.find("] [");
                    ss << line.substr(1, dif - 1);
                    string temp;
                    int num = 0;
                    vector<float> vec;
                    vector<float> out;
                    inputs.push_back(vec);
                    outputs.push_back(out);
                    //vyplni vstup
                    while (!ss.eof()) {
                        ss >> temp;
                        num = stoi(temp);
                        inputs[number_of_inputs].push_back(num);
                    }

                    ss.clear();
                    ss << line.substr(dif + 3, line.find("]", dif + 1) - 4);
                    //vyplni vystup
                    while (!ss.eof()) {
                        ss >> temp;
                        num = stoi(temp);
                        outputs[number_of_inputs].push_back(num);
                    }
                    //zvyseni poctu pripravenych dat
                    number_of_inputs++;


                }
                inputfile.close();
            } else cout << "Unable to open file2\n";
        }

        cout<<"vsetko nacteno pocet hran\n";
        cout << ed<<"\n";

        //topological sorting
        //priprava koeficientov
        vector<T> coeff;
        vector<T> coeffic;
        vector<node> topological;
        int layer=0;
        //pirdanie vstupnej vrstvy
        for(int i=0; i < nodes.size(); i++){
            node nod=nodes[i];
            if(nod.input.size()==0){
                //pridanie vrcholov do usporidania
                topological.push_back(nod);
                nod.layer=layer;
                nodes[i]=nod;
                for(int i=0;i < from.size();i++) {
                    if (from[i] == nod.id) {
                        //vymazanie hran
                        from[i]=-1;
                        to[i]=-1;
                    }
                }
            }
        }
        bool no_input[nodes.size()];
        vector<vector<int>> layers_input;
        //deklarovanie matic
        vector<SparseMatrix<float>> layers_matrix;
        vector<SparseMatrix<float>> layers_ones;
        //deklarovanie listu listov drziacih strukturu grafu
        vector<vector<int>> layers_nodes;
        while(topological.size()!=nodes.size()) {
            //zvacsenie vrstvz
            layer++;
            for (int i = 0; i < nodes.size(); i++) {
                //nastavenie vsetkych booleanov na true aby osm zistil ci je novz vrchol,
                // ktery je prazdnh
                no_input[i] = true;
            }

            for (node nod:topological) {
                //vsetkz vrcholy ktore su v topolgickom usporidani pridame na false
                no_input[nod.id] = false;
            }

            for (int edg:to) {
                // vsetkzm vrcholom ktore maju vstup priradime false
                if(edg!=-1){
                    no_input[edg]=false;
                }
            }

            for (int i=0; i < nodes.size(); i++) {
                //teraz pridame vrcholy bez hran do toplogickeho usporidania spolu s
                if (no_input[i]== true) {
                    node nod = nodes[i];
                    nod.layer=layer;
                    nodes[i]=nod;
                    topological.push_back(nod);
                    //odstranenie hran iducich z pridanych vrcholov
                    for(int j=0;j < from.size();j++){
                        if (from[j] == nod.id) {
                            from[j]=-1;
                            to[j]=-1;
                        }
                    }
                }
            }
        }
        //precislovanie nodov aby boli v rade podla topologicckeho usporiadania
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


        //vztvorenie matic ako approach
        for(int number_of_layer=0;number_of_layer<=layer;number_of_layer++) {
            //pre kazdu vrstvu vytvorime
            vector<T> push_layer;
            vector<int> input_of_layer;
            vector<node> node_of_layer;
            vector<int> id_layer;
            //vytvorenie listu vrcholov a idciek
            for (node nod:nodes) {
                if (nod.layer == number_of_layer) {
                    node_of_layer.push_back(nod);
                    id_layer.push_back(nod.id);
                }
            }
            //vztvorenie input vektoru pre vrstvu
            bool found;
            for (node nod:node_of_layer) {
                for (int i:nod.input) {
                    found = true;
                    //prehladavem ci je uz vrchol pridanz
                    for (int j:input_of_layer) {
                        if (i == j) {
                            found = false;
                            break;
                        }
                    }
                    //ak nie pridame
                    if (found) {
                        input_of_layer.push_back(i);
                    }
                }
            }
            //zotriedi vstup
            sort(input_of_layer.begin(),input_of_layer.end());
            VectorXf bias(node_of_layer.size());
            //deklaracia jednotlivych matic
            SparseMatrix<float> matrix(node_of_layer.size(),input_of_layer.size()+1);
            SparseMatrix<float> ones(node_of_layer.size(),input_of_layer.size()+1);
            //vzmayanie obsahu prechadzajuceho
            coeff.clear();
            coeffic.clear();
            int max = 0;
            for (int i = 0; i < node_of_layer.size(); i++) {
                node nod = node_of_layer[i];
                //prechadzame vsetky vrcholi vo vrstve
                for (int j:nod.input) {
                    //prechadzame input a hladame poradove cislo vstupneho vrchole pre dany vrchol
                    int k = 0;
                    for (int l = 0; l < input_of_layer.size(); l++) {
                        if (input_of_layer[l] == j) {
                            break;
                        }else{
                            k++;
                        }
                    }
                    //pridame do listov vrchol
                    coeff.push_back(T(i,k,dis(gen)));
                    coeffic.push_back(T(i,k,1));

                }
                coeff.push_back(T(i,input_of_layer.size(),1));
                coeffic.push_back(T(i,input_of_layer.size(),1));
            }
            //nasledne listz tripletov dame ako sparse matice a ulozime je
            matrix.setFromTriplets(coeff.begin(),coeff.end());
            ones.setFromTriplets(coeffic.begin(),coeffic.end());
            layers_input.push_back(input_of_layer);
            layers_ones.push_back(ones);
            layers_matrix.push_back(matrix);
            layers_nodes.push_back(id_layer);
        }
        for(int i=0;i<1000;i++) {

            if(generated_input==1) {
                //pripravenie vstupu
                for (int j = 0; j < layers_nodes[0].size(); j++) {
                    results[j] = inputs[i % inputs.size()][j];
                }
                //pripravenie daneho vystupu
                for (int j = 0; j < layers_nodes[number_of_layers - 1].size(); j++) {
                    label[j] = outputs[i % outputs.size()][j];
                }
            }else{
                //pripravenie vstupu
                for (int j = 0; j < layers_nodes[0].size(); j++) {
                    results[j]=dis(gen);
                }
                //pripravenie daneho vystupu
                for (int j = 0; j < layers_nodes[number_of_layers - 1].size(); j++) {
                    label[j] = dis(gen);
                }

            }

            last_entry=1;
            //prejdenie cez vsetky vrstvy
            for (layer = 0; layer < number_of_layers; layer++) {
                vec.resize(layers_input[layer].size()+1);
                //nacitanie vektoru sluziaceho ako vstup pre vrstvu
                for(int j=0;j<layers_input[layer].size();j++){
                    vec(j)=results[layers_input[layer][j]];
                }
                vec(layers_input[layer].size())=1;
                //nasobenie matice vstupom
                res=layers_matrix[layer] * vec;
                //aplikace ativacnej funkcie
                tes=log(1+exp(res.array()));
                //zmenime vysledok do vektoru
                vec = tes.matrix();
                //ulozime vysledok
                layer_results[layer]=tes;
                //this is used cause of the jumping neurons
                //we store the results for each neuron into results and create a new vector everytime for the multiplication
                //ulozenie vysledkov
                for(int j=0;j<vec.size();j++){
                    results[last_entry]=vec(j);
                    last_entry++;
                }
            }
            error=vec-label;
            //spocitanie absolutnej sumy chyb
            loss= error.abs().sum();
            deltas=VectorXf::Zero(nodes.size());
            //backpropagation
            for (layer = number_of_layers-1; layer >=0 ; layer--) {
                //nacti vektor vysledkov
                vec.resize(layers_input[layer].size() + 1);
                for (int j = 0; j < layers_input[layer].size(); j++) {
                    vec(j) = results[layers_input[layer][j]];
                }
                vec(layers_input[layer].size()) = 1;
                //create a delta error

                tes = MatrixXf::Constant(layer_results[layer].size(), 1, 1).array();
                //vypocitanie derivacie aktivacnej funkcie
                tes = tes/(tes+exp(-1*layer_results[layer]));
                //vytvorenie delt
                tes = tes * error;
                error = tes;

                res = error.matrix();

                res = res.transpose() * layers_matrix[layer];
                error = res.array();
                //vytor delty
                for (int j = 0; j < layers_input[layer].size(); j++) {
                    deltas[layers_input[layer][j]] = error(j);
                }
                res = tes.matrix();
                //prenasobeni uciacou rychlostou
                vec = learning_rate * vec;
                //vytovrenie matice updateov
                der = res * vec.transpose();
                //prenasobenie strukturou

                layers_matrix[layer] = layers_matrix[layer] - layers_ones[layer].cwiseProduct(der);
                //uloz delty
                if (layer > 0) {
                    error = VectorXf::Zero(layers_nodes[layer - 1].size());
                    for (int j = 0; j < layers_nodes[layer - 1].size(); j++) {
                        error[j] = deltas[layers_nodes[layer - 1][j]];
                    }
                }
                //need to learn deltas, vec input,tes the delta
                //the equitation for the backprop is delta * times next layer neuron weighted summed together the derivation of activation function times the input
            }
        }


        //cast vzpoctu
        //deklaracia premennych
        VectorXf vec;
        VectorXf res;
        MatrixXf der;
        ArrayXf tes;
        double results[nodes.size()];
        int last_entry=0;
        //vektor vysledkov
        vector<ArrayXf> layer_results;
        int number_of_layers=layers_matrix.size();
        layer_results.resize(number_of_layers);
        //vektor chyb
        ArrayXf error;
        VectorXf label=VectorXf::Constant(layers_nodes[number_of_layers-1].size(),1,5);
        SparseMatrix<double> derivation;
        VectorXf deltas;
        //uciace tempo
        double learning_rate=0.5;
        double loss;
        //

    }
    return 0;
}