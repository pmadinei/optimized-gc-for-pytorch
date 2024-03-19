import objgraph

a = [1,2,3]
a.append(a)

objgraph.show_refs([a], filename='sample-graph.png')