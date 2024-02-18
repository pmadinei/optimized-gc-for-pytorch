# Performance analysis and optimization of Python's cyclic GC in Deep Learning training

## Initial commit
In this repo, we will develop experiments to analyze Python's cyclic garbage collector performance when running Deep Learning (DL) model training workloads in PyTorch. We will research garbage collection algorithms and optimization techniques that can improve performance in the training process of different DL architectures. The project will apply GC configurations and validate their impact by benchmarking and profiling sample DL model training with optimized versus unoptimized configurations. Based on the experimental results, we will quantify garbage collection bottlenecks in training and provide architecture-based guidelines to optimize Python garbage collection for users building and training DL models.

## Week 6 commit
- Explotation and visualization of different object relationships in example Python scripts

Example 1: 
```python
x = []
y = [x, [x], dict(x=x)]
```
![sample-graph](https://github.com/pmadinei/optimized-gc-for-pytorch/assets/45627032/3141312c-2a26-46d7-964c-82547669289e)

-----
Example 2:
```python
class Link:
   def __init__(self, next_link=None):
       self.next_link = next_link

link_3 = Link()
link_2 = Link(link_3)
link_1 = Link(link_2)
link_3.next_link = link_1
A = link_1
```
![sample-cyc-ref](https://github.com/pmadinei/optimized-gc-for-pytorch/assets/45627032/9c7312bd-a529-4c9c-a1c2-64711a7c8c40)

-----
Example 3:
```python
class MyBigFatObject(object):
    pass

def computate_something(_cache={}):

    _cache[42] = dict(foo=MyBigFatObject(),

                      bar=MyBigFatObject())

    # a very explicit and easy-to-find "leak" but oh well

    x = MyBigFatObject() # this one doesn't leak


computate_something()
```
![chain](https://github.com/pmadinei/optimized-gc-for-pytorch/assets/45627032/1907a9f8-5cf4-498b-b4b8-fd7ca980d811)

-----

- Exploration of different data analysis and Machine learning Python scripts for visualizaion and performance profiling
- Exploration of performance profiling packages in Python (deterministic \& static)
