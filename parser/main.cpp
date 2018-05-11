#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Sparse>
#include <algorithm>

using Eigen::MatrixXd;
struct edge{
    int from;
    int to;
    std::string label;
};

struct node{
    int id;
    float value;
    std::vector<float> weights;
    std::string label;
    std::vector<int> input;
    std::vector<int> output;
    int layer;
};

struct graph{
    std::vector<node> nodes;
    std::vector<edge> edges;
};

typedef Eigen::Triplet<double> T;
using namespace Eigen;
using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    string line;
    const clock_t begin_time = clock();
    ifstream myfile ("/mnt/d/parser/graph0.gml");
    vector<node> nodes;
    vector<edge> edges;
    vector<int> from;
    vector<int> to;
    edges.reserve(40000);
    if (myfile.is_open())
    {
        while ( getline(myfile,line) )
        {
            int l=line.find("node");
            int e=line.find("edge");
            if(l > 0){
                node new_node=node();
                getline(myfile,line);
                int id=line.find("id");
                id+=3;
                string substr=line.substr(id, line.size());
                new_node.id=stoi(substr);
                getline(myfile,line);
                id=line.find("label");
                id+=6;
                substr=line.substr(id, line.size());
                new_node.label=substr;
                getline(myfile,line);
                //easy to add node
                nodes.push_back(new_node);
            }
            if(e >0){
                edge newEdge=edge();
                getline(myfile,line);
                int i=line.find("source");
                i+=6;
                newEdge.from=stoi(line.substr(i,line.size()));
                from.push_back(stoi(line.substr(i,line.size())));
                getline(myfile,line);
                i=line.find("target");
                i+=7;
                newEdge.to=stoi(line.substr(i,line.size()));
                to.push_back(stoi(line.substr(i,line.size())));
                getline(myfile,line);
                i=line.find("label");
                i+=6;
                newEdge.label=line.substr(i,line.size());
                edges.push_back(newEdge);
            }

            //string token=line.substr(0,line.find("node"));
            //cout << token << ' ';
        }
        myfile.close();
    }else cout << "Unable to open file";
    for(int i=0;i<nodes.size(); i++ ){
        node nod=nodes[i];
        for(edge edg:edges){
            if(nod.id==edg.to){
                nod.input.push_back(edg.from);
                nod.weights.push_back(0.8);
            }
            if(nod.id==edg.from){
                nod.output.push_back(edg.to);
            }
        }
        nodes[i]=nod;
    }

    edges.clear();


    int input=2;

    //topological sorting

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
    cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    bool no_input[nodes.size()];
    vector<vector<int>> layers_input;
    vector<MatrixXf> layers_matrix;
    vector<VectorXf> layers_biases;
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
    cout<< old_number[483]<<"\n";
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


    VectorXf bias;
    //testing creating a sparse matrix for the first layer using random numbers for initialization

    for(int number_of_layer=1;number_of_layer<=layer;number_of_layer++) {
        cout<<number_of_layer<<"\n";
        vector<T> push_layer;
        vector<int> input_of_layer;
        vector<node> node_of_layer;
        for (node nod:nodes) {
            if (nod.layer == number_of_layer) {
                node_of_layer.push_back(nod);
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
        MatrixXf matrix(node_of_layer.size(),input_of_layer.size());
        for (int i = 0; i < node_of_layer.size(); i++) {
            for (int j = 0; j < input_of_layer.size(); j++) {
                matrix(i,j) = 0;
            }
        }
        int max = 0;
        for (int i = 0; i < node_of_layer.size(); i++) {
            node nod = node_of_layer[i];
            for (int j:nod.input) {
                int k = -1;
                for (int l = 0; l < input_of_layer.size(); l++) {
                    k++;
                    if (input_of_layer[l] == j) {
                        break;
                    }
                }
                if (k > max) {
                    max = k;
                }
                matrix(i,k) = 1;
            }
            bias(i)=0.78248;
        }
        layers_biases.push_back(bias);
        layers_input.push_back(input_of_layer);
        layers_matrix.push_back(matrix);
    }
    //evaluate
    const clock_t begin_time1 = clock();
    VectorXf vec(1);
    VectorXf res;
    ArrayXf tes;
    double results[nodes.size()];
    int last_entry=0;

    vector<ArrayXf> layer_results;
    int number_of_layers=layers_matrix.size();
    layer_results.resize(number_of_layers);
    vector<ArrayXf> error;
    for(int i=0;i<2000;i++) {
        results[0]=0.5;
        last_entry=1;
        for (layer = 0; layer < number_of_layers; layer++) {
            vec.resize(layers_input[layer].size());
            for(int j=0;j<layers_input[layer].size();j++){
                vec(j)=results[layers_input[layer][j]];
            }
            res = layers_matrix[layer] * vec;
            res=res+layers_biases[layer];
            tes = 1/(1+exp(-1*res.array()));
            vec = tes.matrix();
            layer_results[layer]=tes;
            for(int j=last_entry;j<vec.size();j++){
                results[j]=vec(j);
                last_entry++;
            }
        }

        for (layer = number_of_layers-1; layer >=0 ; layer--) {
            vec.resize(layers_input[layer].size());
            for(int j=0;j<layers_input[layer].size();j++){
                vec(j)=results[layers_input[layer][j]];
            }


            tes=MatrixXf::Constant(layer_results[layer].size(),1,1).array();

            tes=tes - layer_results[layer];
            tes=tes*layer_results[layer];
            res=tes.matrix();

        }
    }
    cout << float( clock () - begin_time1 ) /  CLOCKS_PER_SEC <<"\n";
    const clock_t begin_time2 = clock();
    float sum;
    cout<<"test\n";
    for(int j=0;j<2000;j++) {
        for (node nod:nodes) {
            if (nod.input.size() == 0) {
                nod.value = 1;
            } else {
                sum = 0;
                for (int i:nod.input) {
                    sum += nodes[i].value * nod.weights[i];
                }
                nod.value = tanh(sum);
            }
        }
    }
    cout << float( clock () - begin_time2 ) /  CLOCKS_PER_SEC;
    //nacti input random vec staci jednotka na zacatek, pak sprav metodu ktera vznasobz a piecewise vsechny ostatni veci
    //sprav nieco s tym ze vstupasi nebude usporiadany prerobit vstup podla usporiadnia
    return 0;
}