The "RLSNN" folder contains a Microsoft Visual Studio solution file which loads two projects. One is named "Core" which contains the source files of the proposed RL-based SNN and the other one is named "Tester" which contains source files to run the network on an image dataset.
We put a small dataset sampled from Caltech face and motorbike images by which you can examine the code. Please note that the program outputs gnuplot scripts for visualization of features and synaptic weights. You need to install gnuplot if you want to execute them (http://gnuplot.info/).
Besides, depending on your operating system, you need to install the latest Microsoft dotNet framework on Windows, or Mono on Mac/ Linux.

In our paper, we also evaluated CNNs with the same network structure as ours. You can find related python scripts in the "CNN" folder. Those scripts are based on Keras (https://keras.io/) and Tensorflow (https://www.tensorflow.org/).

-------------
Paper details
Title:
First-Spike-Based Visual Categorization Using Reward-Modulated STDP
DOI:
10.1109/TNNLS.2018.2826721