# Privacy-preserving k-means clustering protocol
Based on Jagannathan and Wright's paper:
> http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.114.200&rep=rep1&type=pdf

### Installation
To install required packages, type:

> pip install -r requirements.txt

### Plotting iterations of the algorithm
Set plot to True in naive_kmeans() or secure_kmeans().

Plotted images are created under the images/ directory. 
Please create it before plotting.

### Performance measurement notes
Performance test has been ran on increasingly larger datasets
of 1000 - 32000 elements. The test was run overnight and took
approximately 10 hours to complete, with n_timings=3.
The system specs are as follows:
> CPU: Intel Core i7 6700K @ 4.00GHz
> 
> RAM: 16,0GB Dual-Channel DDR4 @ 1199MHz
> 
> GPU: 4095MB NVIDIA GeForce RTX 2070