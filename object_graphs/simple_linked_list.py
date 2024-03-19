import objgraph

class MyBigFatObject(object):

    pass


def computate_something(_cache={}):

    _cache[42] = dict(foo=MyBigFatObject(),

                      bar=MyBigFatObject())

    # a very explicit and easy-to-find "leak" but oh well

    x = MyBigFatObject() # this one doesn't leak


computate_something()

# import random

# objgraph.show_chain(

#     objgraph.find_backref_chain(

#         random.choice(objgraph.by_type('MyBigFatObject')),

#         objgraph.is_proper_module),

#     filename='chain.png')

# x = []
# y = [x, [x], dict(x=x)]
# import objgraph
# objgraph.show_refs([y], filename='sample-graph.png')

# class Node:
#     def __init__(self, value, left=None, right=None):
#         self.value = value
#         self.left = left
#         self.right = right

# # Create a cycle between nodes
# node1 = Node(1)
# node2 = Node(2)  
# node3 = Node(3)

# node1.right = node2
# node2.left = node1

# # Create a second cycle
# node3.right = node3

# objgraph.show_refs([node3, node2, node1], filename='sample-cyc-ref.png')
import random
class Link:
   def __init__(self, next_link=None):
       self.next_link = next_link

link_3 = Link()
link_2 = Link(link_3)
link_1 = Link(link_2)
link_3.next_link = link_1
A = link_1

objgraph.show_chain(

    objgraph.find_ref_chain(

        link_2,

        objgraph.is_proper_module),

    filename='chain.png')