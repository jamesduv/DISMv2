# DISMv2

Discretization Independent Surrogate Modeling, v2

Introduction
------------
This repository provides supporting code for the manuscript: "Discretization-independent surrogate modeling of physical fields around variable geometries using coordinate-based networks," which is currently under submission and not yet publicly available. 

Tensorflow implementations for the compared methods are provided, including:

1. DV-MLP: Design variable multi-layer perceptron.
2. DV-MLP FF: Design-variable multi-layer-peceptron with Fourier features.
3. DVH: Design variable hypernetwork.
4. DVH-FF: Design variable hypernetworks with Fourier features.

Contents
----------------

The repo should contain the following files:  

-----------------------------------
    DISMv2
    ├── dataloader_poisson.py
    ├── dense_networks.py
    ├── fourier_layers.py    
    ├── hypernet_oneshot_common.py     
    ├── hypernet_oneshot_networks.py
    ├── hypernet_oneshot_train_poisson_dataset.py  
    ├── learning_rate_schedules.py
    ├── problem_settings.py
    ├── tf_util.py
    ├── train_hypernet.py
    ├── train_util.py
    └── README.md
-----------------------------------


**Functionality**

File | Description 
--- | ---|
dense_networks.py | class definition for dense neural network, constructed using Keras sequential API. Used as standalone model for DV-MLP, used as subcomponents for DV-Hnet and NIDS.
keras_hypernetworks.py | class definition for DV-Hnet, subclasses ```tf.keras.Model.```
nids_keras_networks.py | class definition for NIDS, subclasses ```tf.keras.Model.```
tf_common.py | common tensorflow functions



