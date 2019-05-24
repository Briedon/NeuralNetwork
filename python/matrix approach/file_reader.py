from keras.models import Sequential
from keras.layers import Dense
import time
import tensorflow as tf
import numpy as np

graph = {}
for numberOfGraph in range(186):
    f = open("C:/Users/mbriedon/Documents/GitHub/NeuralNetwork/python/graph approach/mutagen/" + str(
        numberOfGraph) + "graph.dot",
             "r")
    f.readline()
    nodes = {}
    edges = {}
    out = {}
    line = ""
    count = 0
    for x in f:
        line = line + x
        if line.__contains__(";"):
            count = count + 1
            parts = line.split("->")
            part1 = parts[0].split("\"")[1].split(" ")[0]
            part2 = parts[1].split("\"")[1].split(" ")[0]
            if part1 in nodes:
                nodes[part1] = nodes[part1] + 1
            else:
                nodes[part1] = 0
            if part2 in nodes:
                nodes[part2] = nodes[part2] + 1
            else:
                nodes[part2] = 0
            if part1 in edges:
                edges[part1].append(part2)
            else:
                edges[part1] = [part2]
            if part2 in out:
                out[part2].append(part1)
            else:
                out[part2] = [part1]
            line = ""
    root = []
    leaves = []
    for n in nodes:
        if n not in edges:
            leaves.append(n)
        if n not in out:
            root.append(n)
    cedges = edges.copy()
    cnodes = nodes.copy()
    level = {}
    lvl = []
    i = 0
    while len(leaves) != 0:
        lvs = []
        for n in leaves:
            level[n] = i
            if n in out:
                lvs.extend(out[n])
        leaves = lvs.copy()
        i = i + 1
    levelArray = {}
    for j in range(i):
        levelArray[j] = []

    for n in nodes:
        levelArray[level[n]].append(n)

    graph[numberOfGraph] = [nodes, edges, out, root, leaves, level, levelArray]
    print(count)

combnodes = {}
combedges = {}
cnodes = 0
lvl = {}
cedges = 0
sedges = 0
snodes = 0
for numberOfGraph in range(16):
    [nodes, edges, out, root, leaves, level, levelArray] = graph[numberOfGraph]
    cnodes = cnodes + len(nodes)
    lvl.update(level)
    for node in nodes:
        if node in combnodes:
            snodes += 1
            if lvl[node] == level[node]:
                combnodes[node].append(numberOfGraph)
                for edg in edges[node]:
                    if edg not in combedges[node]:
                        combedges[node].append(edg)
                        cedges = cedges + 1
                    else:
                        sedges += 1
            else:
                combnodes[node] = [numberOfGraph]
                lvl[node] = level[node]
                for edge in edges:
                    if edge in combedges:
                        combedges[edge].append(edges[edge])
                    else:
                        combedges[edge] = edges[edge]
        else:
            lvl[node] = level[node]
            combnodes[node] = [numberOfGraph]
            for edge in edges:
                combedges[edge] = edges[edge]
maximum = 0
for n in combnodes:
    if maximum < lvl[n]:
        maximum = lvl[n]
maximum += 1
levelArray = [0] * maximum
for j in range(maximum):
    levelArray[j] = []

for n in combnodes:
    levelArray[lvl[n]].append(n)
sizes = [0] * maximum
for j in range(maximum):
    sizes[j] = len(levelArray[j])

dns = []
inp = tf.keras.layers.Dense(sizes[1], input_dim=sizes[0], activation='relu')
dns.append(inp)
for i in range(sizes.__len__() - 1):
    inp = tf.keras.layers.Dense(sizes[i + 1], activation='relu')
    dns.append(inp)
model = tf.keras.models.Sequential(dns)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# for i in range(maximum-1):
#     var=inp[i].get_weights()
#     inp[i].set_weights(matrices[i])
X = np.random.randint(0, 2, [1, sizes[0]])
Y = np.random.randint(0, 2, [1, sizes[sizes.__len__() - 1]])

# st = time.time()
# model.fit(X, Y, epochs=100, batch_size=100)
# print(time.time() - st)


start = time.time()
for i in range(1000):
    model.fit(X, Y, epochs=1, batch_size=1)
print(time.time() - start)
print(len(combnodes) / cnodes)
print(len(combnodes))
