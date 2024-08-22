# IndMask
Implementation of IndMask, an inductive explanation method for multivariate time series black-box models

> IndMask: Inductive Explanation for Multivariate Time Series Black-box Models. Seham Nasr and Sandipan Sikdar. Accepted at ECAI 2024

 ***Please cite our paper in any published work that uses any of these resources.***

 ~~~
 Coming soon
 ~~~

## Abstract

In this paper, we introduce **IndMask**, a framework for explaining decisions of black-box time series models. While there exists a plethora of methods for providing explanations of machine learning models, time series data requires additional considerations. One needs to consider the time aspect in the explanations as well as deal with a large number of input features. Recent work has proposed explaining a time series prediction by generating a mask over the input time series. Each entry in the mask corresponds to an importance score for each feature at each time step. However, these methods only generate instancewise explanations, which means a mask needs to be computed for each input individually, thereby making them unsuited for inductive settings, where explanations need to be generated for numerous inputs, and instancewise explanation generation is severely prohibitive. 
Additionally, these methods have mostly been evaluated on simple recurrent neural networks and are often only applicable to a specific downstream task. 
Our proposed framework **IndMask** addresses these issues by utilizing a parameterized model for mask generation. 
We also go beyond recurrent neural networks and deploy **IndMask** to transformer architectures, thereby genuinely demonstrating its model-agnostic nature.  
The effectiveness of **IndMask** is further demonstrated through experiments over real-world datasets and time series classification and forecasting tasks.
It is also computationally efficient and can be deployed in conjunction with any time series model.

<p align="center"><img src="./IDG.png" width="400" height="400"></p>

## Requirements
