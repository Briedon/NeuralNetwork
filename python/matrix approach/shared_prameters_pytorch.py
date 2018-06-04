import networkx as nx
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


numberOfGraphs=2

graphs={}

for number in range(0,numberOfGraphs):
    #nacitavam graf
    md = nx.read_gml('./graphs/graph' + str(number) + '.gml')
    #vytvorim si prepene ktore pak pouzivam
    # sloncik vstupov jednotlivych nodov
    inputs={}
    #mapovanie na nazvy nodov
    mapping_names={}
    #mapovanie na cisla
    mapping_numbers={}
    i=0
    #vytvorenie mapovania
    for name,id in enumerate(md.nodes()):
        #rozdelenie a zobratie mena z labelu
        nm=id.split("\n")[0]
        #mapujem mena na cele popisy
        mapping_names[id]=nm
        #mapujem cisla na mena
        mapping_numbers[nm]=i
        i += 1
    #prve mapovanie na mena z labelov
    mdnames=nx.relabel_nodes(md,mapping_names)
    #zapisanie vstupov nodov, pre najdenie spolocnych hran
    for name in mdnames.nodes:
        inputs[name]=mdnames.in_edges(name)
    #mapovanie na cisla
    model=nx.relabel_nodes(mdnames,mapping_numbers)
    #pocet vrcholov v danom grafe
    number_of_nodes = int(model.number_of_nodes())
    #inicializacia vstupov
    #dlzka vstupov pre dane vrcholy
    length_of_inputs = [0] * number_of_nodes
    # pocet predchodcov pre dany vrchol
    list_of_predecessors = [-1] * number_of_nodes
    #pole prislusnosti vrcholov k vrstvam
    layer = [0] * number_of_nodes
    #vrcholi
    nodes = list(model.nodes())
    #priradnie vstupov a predchodcov do poli
    for i in range(number_of_nodes):
        length_of_inputs[i] = model.in_degree(nodes[i])
        list_of_predecessors[i] = (list(model.predecessors(nodes[i])))
    #vytvorenie topologick=eho usporiadania
    topological = list(nx.topological_sort(model))
    #cislo vrstvy
    for i in topological:
        # toplogicky prechadzam graf a pridavam vrcholi do vrstiev
        # v ktorych vrstvach je najvacsi spolocni precchodca
        contains = [layer[j] for j in list_of_predecessors[i]]
        if (contains.__len__() == 0):
            layer[i] = 0
            continue
        the_biggest_predecessor = max(contains)
        layer[i] = the_biggest_predecessor + 1
    #vstupna vrstva
    input_layer = {}
    #vrchole vrstvy
    layer_nodes = {}
    #matice
    matrices = {}
    #najvyssia hodnota vrstvy, maxim=alna vrstva
    maximum = max(layer)
    #init velikosti vrstiev
    sizes = [0] * (maximum+1)
    for i in range(maximum+1):
        input_layer[i] = []
        #topologicky prejdeme vrcholi a priradime ktore vrccholy do ktoreho patria
        layer_nodes[i] = [j for j in topological if layer[j] == i]
        sizes[i] = layer_nodes[i].__len__()

    #prejdem vrcholy a pridam ich do vstupov pre jednotlive hrany
    c = 0
    for edge in model.edges():
        a = layer[edge[0]] - 1
        b = layer[edge[1]] - 1
        #kontorla vrstiev a ak je to prehodene tak ich vymenime
        if (a < b):
            a = b
            c = edge[0]
        else:
            c = edge[1]
        not_found = True
        #zistim ci je uz vo vstupnom vektore danej vrstvy
        for i in input_layer[a]:
            if (i == c):
                not_found = False
                break
        #ak este nebol pridany do vektora tak sa prida
        if not_found:
            input_layer[a].extend([c])
    #inicializovanie matic nulami v numpy
    for i in range(maximum):
        matrices[i] = np.zeros((sizes[i + 1], input_layer[i].__len__()))
    #aktualne vytovrenie matic
    for i in range(maximum):
        #nacitanie vrcholov vo vrstve
        nodes = layer_nodes[i + 1]
        next = -1
        for node in nodes:
            next = next + 1
            for edge in model.in_edges(node):
                k = 0
                for j in range(input_layer[i].__len__()):
                    if (edge[0] != input_layer[i][j]):
                        k = k + 1
                    else:
                        break
                matrices[i][next][k] = 1
    #layer_nodes su posunute o jednu vrstvu
    grap=[input_layer,layer_nodes,sizes,matrices,mapping_numbers,inputs,layer,maximum]
    graphs[number]=grap
#spolocne vrcholy
shared_nodes={}
for number in range(0,numberOfGraphs):
    #prejdenie mien vsech grafov a najdenie spolocnyvh vrcholov
    for name in graphs[number][4]:
        if name in shared_nodes.keys():
            shared_nodes[name].append(number)
        else:
            shared_nodes[name]=[number]
graph_shared={}
for number in range(0,numberOfGraphs):
    #cyklus na vytvorenie zdieliacich premennych
    #inicialiyace premennych
    names=graphs[number][4]
    shared={}
    shared_input_in_graphs = {}
    for name in names:
        #temporar list for knowing how much graphs shares the node
        temp=[]
        for graph_number in shared_nodes[name]:
            if graph_number!=number:
                temp.append(graph_number)
        if temp.__len__()!=0:
            shared[name]=temp
    for name in names:
        if name not in shared:
            continue
        serial_name=0
        #id vrcholu
        id=graphs[number][4][name]
        # vrstva zdialeneho vrcholu
        lr=graphs[number][6][id]
        for nam in graphs[number][1][lr]:
            if nam==graphs[number][4][name]:
                break
            else:
                serial_name+=1
        if serial_name==graphs[number][1][lr].__len__():
            print(name)
            print("Chyba1")
        inputs=graphs[number][5][name]
        for input,target in inputs:
            #shname znamena shared name
            #spolocny input, teraz najdem vstupz a pridam im rovnaku hodnotu aby som zachoval rovnake
            shared_input_graphs=[shname for shname in shared_nodes[target] if shname in shared[name]]
            #vyber posledni pouzity graf z hodnot a uloz ho
            chosen_graph=-999999
            chosen_number=-1
            #algoritmus na najdenie posledneho grafu ktory ma spolocnu hranu
            for graph_number in shared_input_graphs:
                if graph_number>number:
                    gr_number=graph_number-numberOfGraphs
                else:
                    gr_number=graph_number
                if gr_number>chosen_graph:
                    chosen_graph=gr_number
                    chosen_number=graph_number
            #cislo vrcholu ktory je zdielany v inom grafe
            number_node=graphs[chosen_number][4][input]
            #cislo vrstvy vo zdielanom grafe
            layer_node=graphs[chosen_number][6][graphs[chosen_number][4][name]]-1
            #poradove cislo vrcholu v liste
            serial_number=0
            #najdenie cisla vrcholu v danom grafe
            for node_number in graphs[chosen_number][0][layer_node]:
                if node_number==number_node:
                    break
                else:
                    serial_number +=1
            if serial_number == graphs[chosen_number][0][layer_node].__len__():
                print(input)
                print("Chyba2")
            serial_number_name=0

            for node_number in graphs[chosen_number][1][layer_node+1]:
                if node_number==graphs[chosen_number][4][name]:
                    break

                else:
                    serial_number_name +=1
            if serial_number_name == graphs[chosen_number][1][layer_node+1].__len__() :
                print(name)
                print("Chyba3")
            serial_node=0
            vrst=graphs[number][6][graphs[number][4][name]]-1
            for node_number in graphs[number][0][vrst]:
                if node_number==graphs[number][4][input]:
                    break
                else:
                    serial_node +=1
            if serial_node == graphs[number][0][vrst].__len__():
                print(input)
                print("Chyba4")
            #serialove cislo mena vo vahach pridam serialove cislo nahrady, vrstva nahrady a vyticeny graf
            shared_input_in_graphs[(serial_name,serial_node,graphs[number][6][graphs[number][4][name]]-1)]=[((serial_number,serial_number_name,layer_node,chosen_number))]
    #ulozime spolocne vrhcoly so vvstunymi hranami, kde dane vstupy su v tuple s
    graph_shared[number]=shared_input_in_graphs

    # tu zacina pytorch
    # linearnu model znazornujuci


class MyLinear(torch.nn.Module):
    # weight je vaha, order je struktura grafu zapisana s jednotkami a bias je offset
    def __init__(self, weight, order, bias):
        """
        v konstruktore inicializujeme premenne, ktore neskor pouzijeme
        pouzivame self lebo budeme je pouyivat mimo funkcie ale stale v triede
        """
        # zdedime valstnosti predka
        super(MyLinear, self).__init__()
        # nastavime vahu
        self.weight = nn.Parameter(weight)
        # bias
        self.bias = nn.Parameter(bias)
        # ulozime ako vyzera vaha
        self.order = order

    def forward(self, x):
        """
        ve forward pouzijeme inicializovane moduly z __init__
        """
        # prenasobenia vah maticou znaciaacou strukturu grafu
        out = self.weight.mul(self.order)
        # hodenie do float hodnoty
        out = out.float()
        # vetkorve prenasobenie matice
        out = torch.mv(out, x)

        out = out + self.bias
        return out
    #dostan ven hodnotu na zdielanie
    def get_value(self):
        return self.weight.clone()

    def set_value(self,weight):
        self.weight=weight
    # list modulov, ktore su potrebne pre pytorch
    # pytorchov autograd neumi delat s polem modulov alebo premmennych


class ListModule(object):
    # mal by fungovat s hocujakym modulom
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    # pridaj dalsi modul do listu rovnako ako v pythone
    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    # funkcia na dlzku
    def __len__(self):
        return self.num_module

    # dostan modul z listu
    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))
    # trida aktualneho modulu


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # definujem si vrstvy ako list modulov pomocou ListModule
        self.layers = ListModule(self, 'layers')
        # vstupna vrstva iba idnetitn=a matica
        self.input_layer = nn.Linear(sizes[0], sizes[0])
        for i in range(maximum):
            # pre kazdu jednu vrstvu si vytvorim
            weight = torch.from_numpy(matrices[i])
            order = Variable(torch.from_numpy(matrices[i], ), requires_grad=False)
            bias = torch.rand(sizes[i + 1])
            # vytovri instanciu novej vrstvy
            layer = MyLinear(weight, order, bias)
            self.layers.append(layer)

    def forward(self, x):
        #zdielanie hodnot
        for node, serial_node, layer in shared_nodes.keys():
            # prejdenie cez vsetky zdielane vrcholy
            serial, serial_number, vrstva, graf = shared_nodes[(node, serial_node, layer)][0]
            we = shared_nets[graf][2].layers[vrstva].get_value()

            wt = self.layers[layer].get_value().data.numpy()

            array=np.full((1),we[serial_number][serial].data[0])
            a = array
            #dosadenie zdielanej hodnoty
            wt[node][serial_node]=a
            self.layers[layer].set_value(nn.Parameter(torch.from_numpy(wt)))

        # kod priebehu neuralnej siete
        # vystupi
        outputs = [0] * number_of_nodes
        # vystup nultej vrstvy
        out = F.relu(self.input_layer(x))
        # zapis nultej vrstvy do
        for j in range(out.__len__()):
            outputs[layer_nodes[0][j]] = out[j]
        for i in range(0, maximum):
            # vytvorenie vstupu do novej matice
            inp = [0] * input_layer[i].__len__()
            for j in range(inp.__len__()):
                # priadnie vysledkov predchadzajucich vsrtiev do vstupu novej
                inp[j] = outputs[input_layer[i][j]]
            inp = torch.cat((inp), 0)
            # zaznamenanie vystupu
            out = F.relu(self.layers[i](inp))

            for j in range(out.__len__()):
                outputs[layer_nodes[i + 1][j]] = out[j]
        # posledny vystup
        return out
shared_nets={}
for j in range(numberOfGraphs):
    net = Net()
    shared_nodes = graph_shared[j]
    input_layer, layer_nodes, sizes, matrices, mapping_numbers, inputs, layer, maximum = graphs[number]
    # deklariacia modelu
    # ak chcem pouzivat gpu potrebujeme
    # net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.00001)
    optimizer.zero_grad()
    criterion = nn.MSELoss()
    shared_nets[j]=[criterion,optimizer,net]

maximum_iterations=2
for iter in range(maximum_iterations):
    for j in range(numberOfGraphs):
        shared_nodes = graph_shared[j]
        criterion, optimizer, net=shared_nets[j]
        target = Variable(torch.rand(sizes[maximum]))
        input = Variable(torch.ones(sizes[0]))
        net.train()
        optimizer.zero_grad()
        out = net(input)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
