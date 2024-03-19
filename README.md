# Performance analysis and optimization of Python's cyclic GC in Deep Learning training

## Initial commit
In this repo, we will develop experiments to analyze Python's cyclic garbage collector performance when running Deep Learning (DL) model training workloads in PyTorch. We will research garbage collection algorithms and optimization techniques that can improve performance in the training process of different DL architectures. The project will apply GC configurations and validate their impact by benchmarking and profiling sample DL model training with optimized versus unoptimized configurations. Based on the experimental results, we will quantify garbage collection bottlenecks in training and provide architecture-based guidelines to optimize Python garbage collection for users building and training DL models.

## Week 6 commit
### Visualized Object Graphs
We used the objgraph module to render object relationships in a Python basic scripts with linked data structures and cycles. Although objgraph graphviz outputs clear cyclic references in simple object references, we can see in example 2 that graphs can get very complex pretty quickly. We still are exploring approaches to assist developers to detect cycle referencing in code during development. We further researched Static Cycle Detection Techniques by reading papers on ahead-of-time cycle detection in scripts and checking approaches that analyze code structure to catch recursive patterns. This would allow optimizing collectors before models are run.

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


## Week 7 commit
### Profiled CNN Training Performance
We developed microbenchmarks with Numpy and pure python to profile the performance of Python's cyclic garbage collector in the convolution process. We then used PyTorch and performed model training with cyclic gc enabled/disabled after each epoch (given the usage of GC by third-party Python packages, adjusting the gc threshold to zero was our approach to completely shutting down the gc). We then compared performance both for CPU and CUDA cores. On our runs, no significant difference in training time and memory usage was observed.

* Cyclic GC Enabled
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        30.98%       11.691s        31.20%       11.772s       1.503ms          7830  
                             aten::convolution_backward        16.04%        6.053s        16.19%        6.111s     390.719us         15640  
                               aten::mkldnn_convolution        14.27%        5.386s        14.49%        5.468s     349.610us         15640  
                          aten::max_pool2d_with_indices        13.46%        5.078s        13.46%        5.078s     324.670us         15640  
                                               aten::mm         5.14%        1.940s         5.14%        1.940s     124.035us         15640  
                               aten::threshold_backward         4.34%        1.636s         4.34%        1.636s     104.594us         15640  
                                            aten::addmm         2.28%     859.037ms         2.42%     911.845ms     116.604us          7820  
                                Optimizer.step#SGD.step         1.62%     612.852ms         3.49%        1.315s     168.165us          7820  
                                            aten::fill_         1.55%     584.556ms         1.55%     584.556ms      18.688us         31280  
                                             aten::add_         1.09%     412.826ms         1.09%     412.826ms       4.400us         93834  
                                        aten::clamp_min         1.03%     390.521ms         1.03%     390.521ms      24.969us         15640
...
Self CPU time total: 37.733s
```

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     234.450ms        14.62%     234.450ms      14.990us           0 b           0 b           0 b           0 b         15640  
void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, fa...         0.00%       0.000us         0.00%       0.000us       0.000us     156.350ms         9.75%     156.350ms      19.994us           0 b           0 b           0 b           0 b          7820  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us     156.280ms         9.75%     156.280ms      19.985us           0 b           0 b           0 b           0 b          7820  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     132.932ms         8.29%     132.932ms       8.499us           0 b           0 b           0 b           0 b         15640  
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us     118.932ms         7.42%     118.932ms       7.605us           0 b           0 b           0 b           0 b         15639  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us     117.212ms         7.31%     117.212ms       7.494us           0 b           0 b           0 b           0 b         15640  
void implicit_convolve_sgemm<float, float, 1024, 5, ...         0.00%       0.000us         0.00%       0.000us       0.000us      92.714ms         5.78%      92.714ms      11.871us           0 b           0 b           0 b           0 b          7810  
cudnn_infer_ampere_scudnn_winograd_128x128_ldg1_ldg4...         0.00%       0.000us         0.00%       0.000us       0.000us      62.832ms         3.92%      62.832ms       8.035us           0 b           0 b           0 b           0 b          7820  
                                ampere_sgemm_128x128_nt         0.00%       0.000us         0.00%       0.000us       0.000us      60.302ms         3.76%      60.302ms       7.711us           0 b           0 b           0 b           0 b          7820  
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us      46.914ms         2.93%      46.914ms       6.000us           0 b           0 b           0 b           0 b          7819  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.300s
Self CUDA time total: 1.604s
```

-----
* Cyclic GC Disabled
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        31.94%       12.062s        32.15%       12.141s       1.551ms          7830  
                             aten::convolution_backward        15.70%        5.927s        15.84%        5.981s     382.437us         15640  
                               aten::mkldnn_convolution        14.17%        5.352s        14.39%        5.434s     347.451us         15640  
                          aten::max_pool2d_with_indices        13.51%        5.103s        13.51%        5.103s     326.249us         15640  
                                               aten::mm         4.89%        1.846s         4.89%        1.846s     118.012us         15640  
                               aten::threshold_backward         4.45%        1.682s         4.45%        1.682s     107.557us         15640  
                                            aten::addmm         2.27%     858.242ms         2.40%     906.348ms     115.901us          7820  
                                Optimizer.step#SGD.step         1.61%     607.067ms         3.47%        1.309s     167.363us          7820  
                                            aten::fill_         1.33%     501.465ms         1.33%     501.465ms      16.031us         31280  
                                             aten::add_         1.10%     416.135ms         1.10%     416.135ms       4.435us         93834
...
Self CPU time total: 37.764s
```

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     234.428ms        14.61%     234.428ms      14.989us           0 b           0 b           0 b           0 b         15640  
void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, fa...         0.00%       0.000us         0.00%       0.000us       0.000us     156.350ms         9.75%     156.350ms      19.994us           0 b           0 b           0 b           0 b          7820  
_ZN19cutlass_cudnn_train6KernelINS_4conv6kernel23Imp...         0.00%       0.000us         0.00%       0.000us       0.000us     156.280ms         9.74%     156.280ms      19.985us           0 b           0 b           0 b           0 b          7820  
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     132.932ms         8.29%     132.932ms       8.499us           0 b           0 b           0 b           0 b         15640  
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us     118.797ms         7.41%     118.797ms       7.596us           0 b           0 b           0 b           0 b         15639  
void at::native::(anonymous namespace)::max_pool_bac...         0.00%       0.000us         0.00%       0.000us       0.000us     117.216ms         7.31%     117.216ms       7.495us           0 b           0 b           0 b           0 b         15640  
void implicit_convolve_sgemm<float, float, 1024, 5, ...         0.00%       0.000us         0.00%       0.000us       0.000us      92.722ms         5.78%      92.722ms      11.872us           0 b           0 b           0 b           0 b          7810  
cudnn_infer_ampere_scudnn_winograd_128x128_ldg1_ldg4...         0.00%       0.000us         0.00%       0.000us       0.000us      62.830ms         3.92%      62.830ms       8.035us           0 b           0 b           0 b           0 b          7820  
                                ampere_sgemm_128x128_nt         0.00%       0.000us         0.00%       0.000us       0.000us      60.649ms         3.78%      60.649ms       7.756us           0 b           0 b           0 b           0 b          7820  
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us      46.914ms         2.92%      46.914ms       6.000us           0 b           0 b           0 b           0 b          7819  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.302s
Self CUDA time total: 1.604s
```
---
## Week 8 commit:
Currently working on:
* Testing GC impact on more complex Pytorch models like multi-modal Transformers
* Experiment with monkey patching to intercept and modify parts of the GC module and object allocation
* Use monkey patching to automatically detect and log cycles during model initialization
* Exploring AST analyzers to detect cycles during code development

## Final commits:
In this final commit, we present the results of our experiments on profiling Python's garbage collector (GC) during the training of deep learning models. We focused on three models: a pure-Python CNN, a PyTorch CNN, and the MDETR vision-language model. We performed multiple runs with different GC configurations, including:
* Calling GC.collect after each training iteration
* Setting the generation 0 threshold to 10
* Setting the generation 0 threshold to 1000
* Disabling the GC entirely (threshold=0)

For each configuration, we measured the average iteration time, the number of collections per generation, and the number of objects collected in each generation.
### Key findings:
* Disabling the GC led to the lowest iteration times across all models. 
* Higher generation 0 thresholds reduced collection frequency and improved performance
* The PyTorch CNN significantly outperformed the pure-Python CNN due to its optimized C/C++ backend
* Figure below shows a comparison of the top 10 tasks (most called functions) in the MDETR model training after 6 iterations, with GC disabled and GC.collect called after each iteration. The execution time for each task is notably higher when GC.collect is called, indicating the overhead introduced by the garbage collector.
  <img width="1443" alt="Figure 1" src="https://github.com/pmadinei/optimized-gc-for-pytorch/assets/45627032/5cb6dee5-ccd9-42da-953e-73437b30d734">

* Figure below illustrates the accumulated difference in execution time between the two GC configurations (disabled and GC.collect called after each iteration) for the MDETR model training. The gap widens as the training progresses, demonstrating the cumulative impact of GC overhead on the overall training time.
  <img width="1489" alt="Figure 2" src="https://github.com/pmadinei/optimized-gc-for-pytorch/assets/45627032/f8defc54-50cc-487e-ac48-83141eb731a6">

* Figure below provides a summary of the execution profile for the MDETR model training with the generation 0 threshold set to 10. The CPU Exec (CPU Execution) dominates the total step time at 79.6%, while other categories, such as Communication, DataLoader, and Other, account for the remaining time. This breakdown helps identify potential bottlenecks and areas for optimization.
<img width="631" alt="Figure 4" src="https://github.com/pmadinei/optimized-gc-for-pytorch/assets/45627032/4305de3b-e21e-457a-b19d-11ed30dda6bd">

* This figure illustrates the same execution summary for the MDETR model training but this time with the generation 0 threshold set to 0 (GC off)
   <img width="631" alt="Figure 3" src="https://github.com/pmadinei/optimized-gc-for-pytorch/assets/45627032/8d956c8a-4442-4c70-9ae3-501bb6c8bd7a">

These results highlight the importance of carefully tuning the garbage collector settings to optimize the performance of deep learning model training in Python. By understanding the impact of GC on different tasks and adjusting the threshold values, developers can significantly reduce training time and improve overall efficiency. Future work will focus on developing more comprehensive guidelines for managing GC across a wider range of architectures and datasets, as well as exploring alternative memory management techniques tailored for deep learning workloads.

Profiling log files can be downloaded [from here](https://drive.google.com/drive/folders/1w1hANyY1frnhQOMG_lPIwCjyZoCS-rPi?usp=sharing). Intructions to open these files are available [here](https://devblogs.microsoft.com/python/python-in-visual-studio-code-february-2021-release/).
