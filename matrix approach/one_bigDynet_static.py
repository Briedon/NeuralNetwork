import networkx as nx
import random
import numpy as np
import dynet_config
#dynet_config.set_gpu()
dynet_config.set(mem=4028)
import dynet as dy

numberOfGraphs=80


#inicializace premennych
graph={}
graph_mapping={}
nodes=0
big_nodes={}
index=0
for number in range(0,numberOfGraphs):
    #nacitanie jednotlivych grafov
    graf = nx.read_gml('./graphs/graph' + str(number ) + '.gml')
    mapping_name={}
    mapping_int={}
    mapping={}
    i=0
    for node in list(graf.nodes()):
        #vystrihnutoe mena z labelu
        separated = node.split("\n")[0]
        nodes=nodes+1
        # namapujem mena na cisla a naopak aby som pak vedel pridat mena k vstupom a vystupom
        mapping[node]=separated
        mapping_int[i]=separated
        mapping_name[separated]=i
        if separated not in big_nodes:
            #ak sa vrchol nenachadza vo velkom grafe tak je pridany
            big_nodes[separated]=index
            index+=1
        i+=1
    graph[number]=nx.relabel_nodes(graf,mapping)
    graph_mapping[number]=[mapping_int,mapping_name]

#vytvorenie modelu velkeho grafu
model = nx.DiGraph()

for name in big_nodes:
    #prdanie vrcholov vo velkom grafe
    model.add_node(big_nodes[name])
print(model.nodes.__len__())
graph_inputs={}
for number in range(0,numberOfGraphs):
    #nacitanie vrcholov grafu
    nodes=graph[number].nodes()
    #nacitanie hran grafu
    edges =list(graph[number].edges())
    for edge in edges:
        #pridam kazdu hranu do noveho modelu
        separated0 = edge[0].split("\n")[0]
        separated1 = edge[1].split("\n")[0]
        #kedze su hrany pridane do mapy(slovniku) tak sa pridaju iba raz cim dosiahneme prepojenie hran
        model.add_edge(big_nodes[separated0],big_nodes[separated1])
    #vytovrenie vstupov pre dane grafy
    mapping_int=graph_mapping[number][0]
    mapping_name=graph_mapping[number][1]
    mod = nx.relabel_nodes(graph[number],mapping_name)
    nodes = list(mod.nodes())
    #list popisujuci vrstvy vrcholov
    layer = [0] * int(mod.number_of_nodes())
    for i in nx.topological_sort(mod):
        contains=[layer[j] for j in list(mod.predecessors(nodes[i]))]
        if(contains.__len__()==0):
            layer[i]=0
            continue
        the_biggest_predecessor=max(contains)
        layer[i]=the_biggest_predecessor+1
    layer_nodes={}
    #ulozim si jednotlive usporiadania vrcholov a velikosti do graph_inputu
    maximum=max(layer)
    sizes=[0]*(maximum+1)
    for i in range(maximum+1):
        layer_nodes[i] = [big_nodes[mapping_int[j]] for j in nodes if layer[j]==i]
        sizes[i]=layer_nodes[i].__len__()
    #takze som si pripravil vrsvty a cisla velikosti pre pridanie vystupov do vytvorenej siete
    #prvy je vstup druhy je vystup oznacene ako vrcholy danych vstupov
    graph_inputs[number]=[layer_nodes[0],layer_nodes[maximum]]
#vymazanie mappingu
graph_mapping.clear()
#vymazanie prebytočných grafov
graph.clear()
print(big_nodes.__len__())
print("number of nodes after grouping")
print(list(model.nodes()).__len__())
#zmenime mena hran na cele cisla aby sa s nimi lepsie manipulovalo
model=nx.relabel_nodes(model,big_nodes)
#okopirovane vytvorenie maticovej reprezentacie pomocou jedneho velkeho grafu
number_of_nodes = int(model.number_of_nodes())
length_of_inputs = [0] * number_of_nodes
list_of_predecessors = [-1] * number_of_nodes
layer = [0] * number_of_nodes
nodes = list(model.nodes())
for i in range(number_of_nodes):
    length_of_inputs[i] = model.in_degree(nodes[i])
    list_of_predecessors[i] = (list(model.predecessors(nodes[i])))

topological = list(nx.topological_sort(model))

layer_number = 0
layer_list = []
for i in topological:
    contains = [layer[j] for j in list_of_predecessors[i]]
    if (contains.__len__() == 0):
        layer[i] = 0
        continue
    the_biggest_predecessor = max(contains)
    layer[i] = the_biggest_predecessor + 1

input_layer = {}
layer_nodes = {}
matrices = {}
maximum = max(layer)
sizes = [0] * (maximum + 1)
for i in range(maximum + 1):
    input_layer[i] = []
    layer_nodes[i] = [j for j in topological if layer[j] == i]
    sizes[i] = layer_nodes[i].__len__()
print("stred")
c = 0
for edge in model.edges():
    a = layer[edge[0]] - 1
    b = layer[edge[1]] - 1
    if (a < b):
        a = b
        c = edge[0]
    else:
        c = edge[1]
    not_found = True
    for i in input_layer[a]:
        if (i == c):
            not_found = False
            break
    if not_found:
        input_layer[a].extend([c])

for i in range(maximum):
    matrices[i]= np.zeros((sizes[i+1],input_layer[i].__len__()))


for i in range(maximum):
    nodes=layer_nodes[i+1]
    next=-1
    for node in nodes:
        next=next+1
        for edge in model.in_edges(node):
            k=0
            for j in range(input_layer[i].__len__()):
                if(edge[0]!=input_layer[i][j]):
                    k=k+1
                else:
                    break
            matrices[i][next][k]=1
print(sizes)
number_of_learning_cycles=2

m = dy.Model()
input = dy.vecInput(sizes[0])
trainer = dy.SimpleSGDTrainer(m)
values = [0] * number_of_nodes
weight = [0] * maximum
bias = [0] * maximum
a = 0

for i in range(maximum):
    if (i == 0):
        a = matrices[i]
        weight[i] = m.add_parameters(a.shape, init=a)
        bias = m.add_parameters((layer_nodes[i + 1].__len__()))
        con = dy.const_parameter(m.add_parameters(a.shape, init=a))
        weight[i] = dy.cmult(weight[i], con)
        result = weight[i] * input
        result = dy.logistic(result + bias)
        for j in range(layer_nodes[i + 1].__len__()):
            values[layer_nodes[i + 1][j]] = result[j]
    else:
        inp = []
        for node in input_layer[i]:
            inp.extend([values[node]])
        a = matrices[i]
        weight[i] = m.add_parameters(a.shape, init=a)
        inp = dy.concatenate(inp)
        bias = m.add_parameters(layer_nodes[i + 1].__len__())
        con = dy.const_parameter(m.add_parameters(a.shape, init=a))
        weight[i] = dy.cmult(weight[i], con)
        result = weight[i] * inp
        result = dy.logistic(result + bias)
        for j in range(layer_nodes[i + 1].__len__()):
            values[layer_nodes[i + 1][j]] = result[j]
y_pred = result
y = dy.vecInput(sizes[maximum])
loss = dy.squared_distance(y_pred, y)

mloss = 0.0
for learning_cycle in range(number_of_learning_cycles):
    for number in range(numberOfGraphs):
        inp=np.zeros(sizes[0])
        for i in range(sizes[0]):
            if layer_nodes[0][i] in graph_inputs[number][0]:
                inp[i]=1
        input.set(np.random.uniform(0, 2, sizes[0]))
        target=y_pred.value()
        for i in range(sizes[maximum]):
            if layer_nodes[maximum][i] in graph_inputs[number][1]:
                target[i]=1
        y.set(target)

        mloss += loss.scalar_value()
        loss.backward()
        trainer.update()


