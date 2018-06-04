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

        double sum=0;
        double learning_rate=0.5;
        for(int j=0;j<10;j++) {
            for (node nod:nodes) {
                //prejdenie vsetkych vrcholov a vypocitanie valstnosti pomocou sumy
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
            //backpropagace
            for (int i=nodes.size()-1;i>=0;i--) {
                node nod=nodes[i];
                //vypocet delty
                if (nod.output.size() == 0) {

                    nod.delta = (dis(gen)-nod.value)*(nod.value)*(1-nod.value);

                } else {
                    nod.delta*=(nod.value)*(1-nod.value);
                }
                for(int k=0;k<nod.input.size();k++){
                    //dopocitanie zmeny hodnot
                    node nd=nodes[nod.input[k]];
                    nd.delta+=nod.delta*nod.weights[k]*learning_rate;
                    nod.weights[k]=nod.delta*nod.input[k];
                    nodes[nod.input[k]]=nd;
                }
                //vratnie vrcholu naspat s upravenymi valstnstami
                nodes[i]=nod;
            }
        }


    }
    return 0;
}