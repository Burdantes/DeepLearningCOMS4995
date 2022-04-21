#  DeepLearning COMS4995 - Comparative Analysis of Recommender Systems: Impact on Bias and Polarization

As stated in the introduction, a key of our contribution is to ensure the replicability of our study and permit to reiterate the study as easily as possible. The core of our code is accessible on the Github repository on a Jupyter notebook named FinalNotebook.ipynb, it is decomposed in a couple of steps. It relies on a couple of libraries that are explictly stated in the \verb|requirements.txt| and we explain how to create a virtual environment in the \verb|README.md|. The first portion of the notebook illustrates how we generate our synthetic data. The cells are thoroughly commented to ensure easy reproducibility and flexibility on usage. In the notebook, our plot focuses on the case where our content popularity follows a Zipf law as it has been proven to be an appropriate distribution to model content popularity in the past (by opposition to the uniformly distributed case of items in Figure 2). The second portion of our Jupyter notebook investigates statistics of both the MovieLens data and our synthetically generated data. This demonstrates that our synthetic data exhibits similar properties as the MovieLens data making it a satisfactory depiction of reality. The second portion of our script illustrates how we can leverage the different libraries to build a simple ALS recommender system. The idea here is to illustrate the different components that are hidden in the third portion of the benchmarking. The output from that recommendation can be then leveraged to get more intuition on the recommendation. The third portion is as stated earlier the benchmarking script. It corresponds of an automated cell that trains 7 recommender systems on MovieLens and computes a batch of statistics for every subcategories. Note that we only focus on age and gender as we had restricted time but the script has been built with other meta-information in mind and could be easily extended as such. It also can be applied to synthetically generated data and we might train it for that purpose for the Sunday deadline time permiting. The last step of our script constitutes the graph generation portion and allows to recreate the graph obtained in Figure \ref{fig:recommendation}. The graph visualisation itself has not been integrated as it relies on external softwares (namely Gephi and Adobe Illustrator).  

## How to add recommender systems of interest (to be completed before Sunday)

We also explain how other recommender systems of interest can be integrated painlessly in the README.md|.

## Creating virtual environment and downloading libraries

Create a virtual environment and activate it to run the python 3 scripts:
```bash
$ python3 -m venv ./DeepLearningCOMS4995
$ source ./DeepLearningCOMS4995/bin/activate
(./DeepLearningCOMS4995)$ pip3 install -r requirements.txt
```
All the code can be found in the ``FinalNotebook.ipynb`` file and that script has been documented to be self-sufficient. 

## Acknowledgements
We owe a lot to the authors of the following libraries and notebooks:
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Microsoft recommenders](https://github.com/microsoft/recommenders/)
- [Tsinghua University GNN recommender systems](https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems)
- [Awesome-Rec](https://github.com/RUCAIBox/Awesome-RSPapers)
- [KDD tutorial on Recommender Systems](https://sites.google.com/view/kdd20-marketplace-autorecsys/)
- 