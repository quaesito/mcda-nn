# mcda-nn

How often does it happen that you have to deal with structured data and having to rank your ‘options’? 

Even more difficultly, how would you tackle this problem if your data was incomplete and you can use only the little data you have to create forecasts of rankings?

In this repo, I will share with you my experience towards an iterative process which led to a systematic and satisfactory approach to rank options. Herein, I show you one example applied to a biofuel-related problem and dataset, however the approach is generalizable to other types of datasets.
The approach involves the usage of Multi-Criteria Decision Analyses, including Weighted Sum Model (WSM), Weighted Product Model (WPM) and Topsis to produce ranking of decisions.
Subsequently, Multi-variate Regression, Deep Neural Network (DNN) and a Multi-layer Perceptron (MLP) are trained to predict such rankings. The results are duly compared and discussed.

This repo is associated with the following medium story
https://towardsdatascience.com/training-neural-networks-to-predict-rankings-8a3308c472e6
