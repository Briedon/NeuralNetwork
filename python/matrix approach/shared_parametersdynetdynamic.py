import networkx as nx
import random
import numpy as np
import dynet_config
#dynet_config.set_gpu()
dynet_config.set(mem=3200)
import dynet as dy



numberOfGraphs=186

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
shared_models={}

m = dy.Model()
def create_graph(init_weight, init_con, init_bias, weight, bias, con, input, target,number,nodes):
    input_layer, layer_nodes, sizes, matrices, mapping_numbers, inputs, layer, maximum = graphs[number]
    values = [0] * nodes
    dy.renew_cg()
    shared_nodes=graph_shared[number]
    #node je srialove cislo a layer je vrstva
    for node,serial_node,layer in shared_nodes.keys():
        #prejdenie cez vsetky zdielane vrcholy
        serial,serial_number,vrstva,graf=shared_nodes[(node,serial_node, layer)][0]
        we=shared_models[graf][3][vrstva].value()

        wt=weight[layer].value()
        wt[node][serial_node]=we[serial_number][serial]
        weight[layer].set_value(wt)
    x = dy.vecInput(input.size)
    x.set(input)
    y = dy.vecInput(target.size)
    y.set(target)
    result = init_weight * x
    hgt = dy.cmult(init_weight, init_con)
    init_weight.set_value(hgt.value())
    result = dy.logistic(result + init_bias)
    for j in range(layer_nodes[0].__len__()):

        values[layer_nodes[0][j]] = result[j]
    for i in range(maximum):
        inp = []
        for node in input_layer[i]:
            inp.extend([values[node]])
        inp = dy.concatenate(inp)
        weight[i].set_value(dy.cmult(weight[i], con[i]).value())
        result = weight[i] * inp
        result = dy.logistic(result + bias[i])
        for j in range(layer_nodes[i + 1].__len__()):
            values[layer_nodes[i + 1][j]] = result[j]
    loss = dy.squared_distance(y, result)
    return loss
#vygenerovanie grafov
for number in range(numberOfGraphs):
    input_layer, layer_nodes, sizes, matrices, mapping_numbers, inputs, layer, maximum=graphs[number]
    input = dy.vecInput(sizes[0])
    values = [0] * layer.__len__()
    #inicializace nultej vrstvy
    init_weight = m.add_parameters((sizes[0], sizes[0]), init='identity')
    init_con = m.add_parameters((sizes[0], sizes[0]), init='identity')
    init_bias = m.add_parameters((sizes[0]))
    #pripravenie listov na zapis a pridanie do parametrov modelu
    weight = [0] * maximum
    con = [0] * maximum
    bias = [0] * maximum
    a = 0

    for i in range(maximum):
        #inicializacia vah
        a = matrices[i]
        #pridas parameter plus urcis rozmer a zaciatok
        weight[i] = m.add_parameters(a.shape, init=a)
        bias[i] = m.add_parameters((layer_nodes[i + 1].__len__()))
        con[i] = m.add_parameters(a.shape, init=a)
    #nastavenie metody na gradientni zostup
    trainer = dy.SimpleSGDTrainer(m)
    #zdielany model
    shared_models[number]=[init_weight, init_con,init_bias,weight,bias,con,np.random.uniform(0,2,sizes[0]),np.random.uniform(0,2,sizes[maximum]),layer.__len__(),trainer]
    dy.renew_cg()
for iter in range(100):
    for number in range(numberOfGraphs):
        #nactu zdialane informacie z bunky
        init_weight, init_con, init_bias, weight, bias, con,input, output,nodes, trainer =shared_models[number]
        mloss = 0.0
        #spustim jeden prechod grafu so zdielanimi bunkami
        loss = create_graph(init_weight, init_con, init_bias, weight, bias, con, input,
                            output,number,nodes)
        loss.backward()
        trainer.update()
