# DeepFM--Factorization-supported-neural-network
factorization machine and Feed Forward neural network with embedding layer for CTR
Please read this paper first to understand:
""DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
by Huifeng Guo*2, Ruiming Tang2, Yunming Ye†1, Zhenguo Li2, Xiuqiang He2""
https://www.ijcai.org/proceedings/2017/0239.pdf
Information About The file;
1.DataReader- It take the data and convert to the formate needed for embedding layers used by file Main
2.Main- It's the most imortant file which contain all the preprocesseing, test train split of the data and parameters needed..
3.DeepFM- this file does all the background work of machine learning from tensorflow's bulding tensors for complete optimization techiniques, like working with loss funtion, evaluation matrix and all
4.Matrix- this in not that important and use of it depends upon your need to bulid your own evaluation matrix
5.config- Just open it it's just the address file for your data directory in your device
