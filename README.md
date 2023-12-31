# KNNSampler
KNNSampler is an implementation of the [Research paper](https://ieeexplore.ieee.org/document/8990391). It is created to help developers reduce the size of their Datasets by sampling the "Representatives" from the same. NN_SCORES and MNN_SCORES, as discussed in the referred paper, were used to find these "Representatives". KNNSampler works in both dynamic and static way, as discussed by the author in the paper.
### Setup
- Python 3.10
- Requirenments : numpy, pandas, sklearn
### Enhancements
- MNN_SCORES are calculated after every iteration for the entire dataset in the algorithm suggested in the research paperwhich. This leads to redundant calculations. Hence, in this package we only calculate MNN_SCORES for the shortlisted rows using NN_SCORES, producing the same result as the original algorithm but in an optimal way.
- Error was found in the line : train sample = train sample ∪ X[index] in the algorithm given in the research paper, we replace X[index] with X[train_index] for correct outcome.
- Error was found in the Until loop logic of algorithm in the research paper : (NN − score(X) = 0) ∨ (| train sample |≤ k); The second condition must be |X| <= k, changes were done.
- Values of t, m, s for (t,m,s)-nets were not provided in the paper, We give users the freedom to choose the t, m, and s values or use the default values provided.
### Important
- The dataset passed to the sample() function must **NOT CONTAIN COLUMN NAMED "idx"**.
- Warnings produced by "drop()" function in pandas.DataFrame must be **IGNORED**, since they have been added for debug purposes.
### Navigate
- [Package](https://github.com/snimale/KNNSampler/tree/dev/KNearestNeighborSampling)
- [Example Usecase](https://github.com/snimale/KNNSampler/tree/dev/others/example-usecase)
- [Example Results](https://github.com/snimale/KNNSampler/tree/dev/others/sampled_data_plotted_results)
- [Sratch Code](https://github.com/snimale/KNNSampler/tree/dev/others/knn-sampling-scratch-code)

### Acknowledgement
I have "implemented" and "added optimizations" to the original research work done by : Bheekya Dharamsotu, K. Swarupa Rani, Salman Abdul Moiz, and C. Raghavendra Rao in the research paper : </br> </br>
B. Dharamsotu, K. S. Rani, S. Abdul Moiz and C. R. Rao, "k-NN Sampling for Visualization of Dynamic Data Using LION-tSNE," 2019 IEEE 26th International Conference on High Performance Computing, Data, and Analytics (HiPC), Hyderabad, India, 2019, pp. 63-72, doi: 10.1109/HiPC.2019.00019.
