import objgraph

class MyBigFatObject(object):

    pass


def computate_something(_cache={}):

    _cache[42] = dict(foo=MyBigFatObject(),

                      bar=MyBigFatObject())

    # a very explicit and easy-to-find "leak" but oh well

    x = MyBigFatObject() # this one doesn't leak


computate_something()

import random

objgraph.show_chain(

    objgraph.find_backref_chain(

        random.choice(objgraph.by_type('MyBigFatObject')),

        objgraph.is_proper_module),

    filename='chain.png') 

roots = objgraph.get_leaking_objects()

len(roots)

objgraph.show_most_common_types(objects=roots)

objgraph.show_refs(roots[:3], refcounts=True, filename='roots.png')