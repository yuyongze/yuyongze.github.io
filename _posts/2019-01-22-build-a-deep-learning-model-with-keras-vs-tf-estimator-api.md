---
layout: post
published: true
title: Build a Deep Learning Model with Keras vs tf.estimator API
date: '2019-01-22'
tags:
  - Data Science
  - Deep Learning
---
# Build a Deep Learning Model with Keras vs tf.estimator API

Here we will use [MNIST](https://www.tensorflow.org/tutorials/) dataset to show how to make classification model using [Keras](https://keras.io/) and [tf.estimator API](https://www.tensorflow.org/api_docs/python/tf/estimator). Let's compare with those two models and at last I will dicuss about my opinion about those two.

There 4 section included :

- [Load dataset](#section1)
- [Keras model](#section2)
- [tf.estimator API](#section3)
- [tf.estimator on large dataset](#section4)

#### update tensorflow to the newest version

```python
!pip install --upgrade tensorflow 
```


#### import libraries

```python
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Flatten,Dense,Dropout
from keras.models import Sequential
import numpy as np
import pandas as pd
tf.__version__
```

```
Using TensorFlow backend.
```





```
'1.12.0'
```



# <a name="section1"></a> Load data

```python
(x_train, y_train),(x_test, y_test) = mnist.load_data()
```

```
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
```

#### inpect data

Let's how many examples in training set and testing set. And each picture contains 28*28 pixels.

```python
print ("Traning set feature shape: ", x_train.shape,"\n")
print ("Traning set label shape: ", y_train.shape,"\n")
print ("Testing set feature shape: ", x_train.shape,"\n")
print ("Testing set feature shape: ", y_train.shape,"\n")
```

```
Traning set feature shape:  (60000, 28, 28) 

Traning set label shape:  (60000,) 

Testing set feature shape:  (60000, 28, 28) 

Testing set feature shape:  (60000,) 
```



```python
y_train[0:10]
```



```
array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4], dtype=uint8)
```



So features contian 28*28 pixals  0-255 grey scale,

label is list of number 0-9. Now let's normalize feature between 0 and 1

```python
# noramlized features from 0-1  
x_train, x_test = x_train / 255.0, x_test / 255.0
```

```python
# define input_size
dim=28
```

# <a name="section2"></a> Keras model

lets run a baby example using keras

*Sequential* is defined in [Keras.layers](https://keras.io/layers/about-keras-layers/) module, once we define **"model=Sequential()"**. We then could used model.add() function to add any layers to define the model complexity.  
One another thing need to notice: the first layer need a **input_shape** arugement to define the input tensor shape
How easy it is?!   
The last line of code **"model.summary()"** return a table of parameters of the model you just build.

```python
model = Sequential()
model.add(Flatten(input_shape=(dim,dim,)))
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))
model.summary()
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

Once the model has been defined, we need [compile](https://keras.io/models/model/#compile) the model and pass arguments like *optimizer*, *loss* and *metrics*.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

Then use **fit** function and pass **features tensors(or numpy arrays)** and **labels**. 

```python
model.fit(x_train, y_train, batch_size= 32, epochs=5)
```

```
Epoch 1/5
60000/60000 [==============================] - 93s 2ms/step - loss: 0.2188 - acc: 0.9351
Epoch 2/5
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0975 - acc: 0.9707
Epoch 3/5
60000/60000 [==============================] - 102s 2ms/step - loss: 0.0675 - acc: 0.9788
Epoch 4/5
60000/60000 [==============================] - 109s 2ms/step - loss: 0.0539 - acc: 0.9825
Epoch 5/5
60000/60000 [==============================] - 95s 2ms/step - loss: 0.0440 - acc: 0.9858
```





```
<keras.callbacks.History at 0x7f7966ae4898>
```



```python
model.evaluate(x_test, y_test)
```

```
10000/10000 [==============================] - 3s 307us/step
```





```
[0.068943760562455284, 0.9788]
```



It is returning the loss and metrics we difined before.  
I got accuracy: 0.9788 on my eval set on 5 epochs from scrach.

# <a name="section3"></a> tf.estimator API

#### DNNClassifier estimator API 

the following code in this section is modified from a [blog](https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940) by *Macro Lanaro*.

To build estimator api, it needs to define a *"input function"* including *"feature columns"* and *"classifier"*. 

```python
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_train},
    y=y_train.astype(np.int32),
    num_epochs=None,
    batch_size=50,
    shuffle=True
)
```

**tf.feature_columns** is a powerful tool to define different type of features. Use *feature_columns* function, we can easily define different type of columns like **bucketized_column, categorical_column_with_hash_bucket, categorical_column_with_identity**, which means if the dataset needs feature engineering before training, this would be a good choice.  (See other *feature_columns* [here](https://www.tensorflow.org/api_docs/python/tf/feature_column))

```python
# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]
```

There are some pre-defined models(Regressor and Classifier) in estimator API. I will list some and give some short explanations here:  
[BaselineClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/BaselineClassifier): This classifier ignores feature values and will learn to predict the average value of each label. For single-label problems, this will predict the probability distribution of the classes as seen in the labels. For multi-label problems, this will predict the fraction of examples that are positive for each class. (**It basically do nothing.**)  
[BoostedTreesClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesClassifier): A Classifier for Tensorflow Boosted Trees models. (I have not used TFBT yet, but If you want to know more about TFBT , you can find some sources [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/boosted_trees) and [there](http://ecmlpkdd2017.ijs.si/papers/paperID705.pdf))  
[DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier): A classifier for TensorFlow DNN models. (**It is easy to define a simple DNN model by using like "hidden_units=[1024, 512, 256]"**)  
[DNNLinearCombinedClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier): An estimator for TensorFlow Linear and DNN joined classification models. (**This is more costomized model. If some of columns you want to use Linear and some of the columns DNN, this model would be great. "linear_feature_columns=[...], dnn_feature_columns=[..]"**)  
[LinearClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier):Train a linear model to classify instances into one of multiple possible classes. When number of possible classes is 2, this is binary classification. (**It is a Logisticregression with binary or Softmax layer with multiclasses, simple and easy**)



```python
# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.2,
    model_dir="./tmp/mnist_model"
)

```

```
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_save_summary_steps': 100, '_save_checkpoints_steps': None, '_global_id_in_cluster': 0, '_task_id': 0, '_task_type': 'worker', '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f780c5a7278>, '_num_worker_replicas': 1, '_protocol': None, '_eval_distribute': None, '_master': '', '_is_chief': True, '_tf_random_seed': None, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_num_ps_replicas': 0, '_keep_checkpoint_every_n_hours': 10000, '_save_checkpoints_secs': 600, '_train_distribute': None, '_experimental_distribute': None, '_model_dir': './tmp/mnist_model', '_log_step_count_steps': 100, '_device_fn': None, '_keep_checkpoint_max': 5, '_service': None, '_evaluation_master': ''}

```



```python
import shutil
shutil.rmtree("./tmp/mnist_model", ignore_errors = True)
classifier.train(input_fn=train_input_fn, steps=30000)

```

```
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into ./tmp/mnist_model/model.ckpt.
INFO:tensorflow:loss = 121.802, step = 1
INFO:tensorflow:global_step/sec: 61.4317
INFO:tensorflow:loss = 68.5095, step = 101 (1.702 sec)
INFO:tensorflow:global_step/sec: 62.6354
INFO:tensorflow:loss = 49.4702, step = 201 (1.514 sec)
INFO:tensorflow:global_step/sec: 70.9626
INFO:tensorflow:loss = 37.7748, step = 301 (1.405 sec)
INFO:tensorflow:global_step/sec: 67.2957
INFO:tensorflow:loss = 31.825, step = 401 (1.485 sec)
INFO:tensorflow:global_step/sec: 52.5849
INFO:tensorflow:loss = 28.4717, step = 501 (1.902 sec)
INFO:tensorflow:global_step/sec: 62.6429
INFO:tensorflow:loss = 21.2868, step = 601 (1.596 sec)
INFO:tensorflow:global_step/sec: 62.5158
INFO:tensorflow:loss = 33.1921, step = 701 (1.599 sec)
INFO:tensorflow:global_step/sec: 62.3679
INFO:tensorflow:loss = 17.8847, step = 801 (1.604 sec)
INFO:tensorflow:global_step/sec: 71.6003
INFO:tensorflow:loss = 27.3183, step = 901 (1.397 sec)
INFO:tensorflow:global_step/sec: 62.0781
INFO:tensorflow:loss = 18.1688, step = 1001 (1.611 sec)
INFO:tensorflow:global_step/sec: 66.5458
INFO:tensorflow:loss = 19.8525, step = 1101 (1.503 sec)
INFO:tensorflow:global_step/sec: 62.9073
INFO:tensorflow:loss = 19.2064, step = 1201 (1.590 sec)
INFO:tensorflow:global_step/sec: 58.811
INFO:tensorflow:loss = 37.9015, step = 1301 (1.700 sec)
INFO:tensorflow:global_step/sec: 62.5433
INFO:tensorflow:loss = 15.6147, step = 1401 (1.599 sec)
INFO:tensorflow:global_step/sec: 62.4846
INFO:tensorflow:loss = 16.1482, step = 1501 (1.601 sec)
INFO:tensorflow:global_step/sec: 66.0724
INFO:tensorflow:loss = 13.6277, step = 1601 (1.593 sec)
INFO:tensorflow:global_step/sec: 59.1684
INFO:tensorflow:loss = 20.6206, step = 1701 (1.610 sec)
INFO:tensorflow:global_step/sec: 71.0037
INFO:tensorflow:loss = 18.4882, step = 1801 (1.409 sec)
INFO:tensorflow:global_step/sec: 66.718
INFO:tensorflow:loss = 18.6254, step = 1901 (1.499 sec)
INFO:tensorflow:global_step/sec: 71.2125
INFO:tensorflow:loss = 24.3903, step = 2001 (1.404 sec)
INFO:tensorflow:global_step/sec: 62.8458
INFO:tensorflow:loss = 13.7387, step = 2101 (1.591 sec)
INFO:tensorflow:global_step/sec: 62.2757
INFO:tensorflow:loss = 10.8889, step = 2201 (1.606 sec)
INFO:tensorflow:global_step/sec: 55.6268
INFO:tensorflow:loss = 14.1414, step = 2301 (1.798 sec)
INFO:tensorflow:global_step/sec: 62.5372
INFO:tensorflow:loss = 17.201, step = 2401 (1.599 sec)
INFO:tensorflow:global_step/sec: 62.8682
INFO:tensorflow:loss = 21.4618, step = 2501 (1.591 sec)
INFO:tensorflow:global_step/sec: 66.0703
INFO:tensorflow:loss = 22.8158, step = 2601 (1.513 sec)
INFO:tensorflow:global_step/sec: 59.2054
INFO:tensorflow:loss = 15.5993, step = 2701 (1.689 sec)
INFO:tensorflow:global_step/sec: 62.5741
INFO:tensorflow:loss = 8.24551, step = 2801 (1.600 sec)
INFO:tensorflow:global_step/sec: 66.7778
INFO:tensorflow:loss = 9.84984, step = 2901 (1.496 sec)
INFO:tensorflow:global_step/sec: 76.5505
INFO:tensorflow:loss = 11.4913, step = 3001 (1.306 sec)
INFO:tensorflow:global_step/sec: 71.0544
INFO:tensorflow:loss = 12.5438, step = 3101 (1.408 sec)
INFO:tensorflow:global_step/sec: 62.8107
INFO:tensorflow:loss = 14.3298, step = 3201 (1.595 sec)
INFO:tensorflow:global_step/sec: 66.9391
INFO:tensorflow:loss = 18.0646, step = 3301 (1.491 sec)
INFO:tensorflow:global_step/sec: 54.992
INFO:tensorflow:loss = 8.15245, step = 3401 (1.819 sec)
INFO:tensorflow:global_step/sec: 67.029
INFO:tensorflow:loss = 10.1809, step = 3501 (1.492 sec)
INFO:tensorflow:global_step/sec: 67.2156
INFO:tensorflow:loss = 5.40207, step = 3601 (1.488 sec)
INFO:tensorflow:global_step/sec: 65.9882
INFO:tensorflow:loss = 14.5899, step = 3701 (1.515 sec)
INFO:tensorflow:global_step/sec: 62.8134
INFO:tensorflow:loss = 13.0695, step = 3801 (1.595 sec)
INFO:tensorflow:global_step/sec: 16.9465
INFO:tensorflow:loss = 16.3003, step = 3901 (5.898 sec)
INFO:tensorflow:global_step/sec: 62.7308
INFO:tensorflow:loss = 16.1697, step = 4001 (1.594 sec)
INFO:tensorflow:global_step/sec: 70.9951
INFO:tensorflow:loss = 5.55165, step = 4101 (1.410 sec)
INFO:tensorflow:global_step/sec: 66.228
INFO:tensorflow:loss = 7.96833, step = 4201 (1.587 sec)
INFO:tensorflow:global_step/sec: 56.1865
INFO:tensorflow:loss = 8.52587, step = 4301 (1.701 sec)
INFO:tensorflow:global_step/sec: 66.4251
INFO:tensorflow:loss = 6.85529, step = 4401 (1.505 sec)
INFO:tensorflow:global_step/sec: 58.7
INFO:tensorflow:loss = 10.1988, step = 4501 (1.704 sec)
INFO:tensorflow:global_step/sec: 58.9096
INFO:tensorflow:loss = 8.05029, step = 4601 (1.698 sec)
INFO:tensorflow:global_step/sec: 66.6605
INFO:tensorflow:loss = 14.5497, step = 4701 (1.500 sec)
INFO:tensorflow:global_step/sec: 66.7875
INFO:tensorflow:loss = 21.4583, step = 4801 (1.497 sec)
INFO:tensorflow:global_step/sec: 71.282
INFO:tensorflow:loss = 9.96876, step = 4901 (1.405 sec)
INFO:tensorflow:global_step/sec: 71.2905
INFO:tensorflow:loss = 18.527, step = 5001 (1.403 sec)
INFO:tensorflow:global_step/sec: 66.8843
INFO:tensorflow:loss = 10.306, step = 5101 (1.493 sec)
INFO:tensorflow:global_step/sec: 66.8142
INFO:tensorflow:loss = 6.91562, step = 5201 (1.497 sec)
INFO:tensorflow:global_step/sec: 55.2372
INFO:tensorflow:loss = 7.09249, step = 5301 (1.813 sec)
INFO:tensorflow:global_step/sec: 62.6079
INFO:tensorflow:loss = 6.97725, step = 5401 (1.595 sec)
INFO:tensorflow:global_step/sec: 62.4493
INFO:tensorflow:loss = 10.8742, step = 5501 (1.689 sec)
INFO:tensorflow:global_step/sec: 66.6577
INFO:tensorflow:loss = 3.34514, step = 5601 (1.415 sec)
INFO:tensorflow:global_step/sec: 77.119
INFO:tensorflow:loss = 11.7359, step = 5701 (1.295 sec)
INFO:tensorflow:global_step/sec: 62.2129
INFO:tensorflow:loss = 7.89293, step = 5801 (1.693 sec)
INFO:tensorflow:global_step/sec: 62.8196
INFO:tensorflow:loss = 5.07294, step = 5901 (1.505 sec)
INFO:tensorflow:global_step/sec: 58.9503
INFO:tensorflow:loss = 8.46785, step = 6001 (1.696 sec)
INFO:tensorflow:global_step/sec: 71.2018
INFO:tensorflow:loss = 5.75749, step = 6101 (1.404 sec)
INFO:tensorflow:global_step/sec: 62.1246
INFO:tensorflow:loss = 10.8719, step = 6201 (1.610 sec)
INFO:tensorflow:global_step/sec: 66.9153
INFO:tensorflow:loss = 7.60483, step = 6301 (1.584 sec)
INFO:tensorflow:global_step/sec: 62.7724
INFO:tensorflow:loss = 5.19591, step = 6401 (1.503 sec)
INFO:tensorflow:global_step/sec: 55.6651
INFO:tensorflow:loss = 7.07051, step = 6501 (1.798 sec)
INFO:tensorflow:global_step/sec: 62.2735
INFO:tensorflow:loss = 14.4383, step = 6601 (1.605 sec)
INFO:tensorflow:global_step/sec: 58.8694
INFO:tensorflow:loss = 8.78651, step = 6701 (1.699 sec)
INFO:tensorflow:global_step/sec: 66.1878
INFO:tensorflow:loss = 7.1201, step = 6801 (1.510 sec)
INFO:tensorflow:global_step/sec: 71.8224
INFO:tensorflow:loss = 5.48161, step = 6901 (1.392 sec)
INFO:tensorflow:global_step/sec: 66.7632
INFO:tensorflow:loss = 11.9077, step = 7001 (1.498 sec)
INFO:tensorflow:global_step/sec: 66.604
INFO:tensorflow:loss = 4.50428, step = 7101 (1.501 sec)
INFO:tensorflow:global_step/sec: 62.6609
INFO:tensorflow:loss = 12.5488, step = 7201 (1.599 sec)
INFO:tensorflow:global_step/sec: 66.0236
INFO:tensorflow:loss = 14.973, step = 7301 (1.591 sec)
INFO:tensorflow:global_step/sec: 66.8932
INFO:tensorflow:loss = 5.47284, step = 7401 (1.415 sec)
INFO:tensorflow:global_step/sec: 62.6644
INFO:tensorflow:loss = 14.1632, step = 7501 (1.596 sec)
INFO:tensorflow:global_step/sec: 66.49
INFO:tensorflow:loss = 10.5252, step = 7601 (1.504 sec)
INFO:tensorflow:global_step/sec: 70.3663
INFO:tensorflow:loss = 4.26116, step = 7701 (1.421 sec)
INFO:tensorflow:global_step/sec: 59.8709
INFO:tensorflow:loss = 3.39557, step = 7801 (1.671 sec)
INFO:tensorflow:global_step/sec: 55.65
INFO:tensorflow:loss = 2.8155, step = 7901 (1.796 sec)
INFO:tensorflow:global_step/sec: 58.8193
INFO:tensorflow:loss = 2.47339, step = 8001 (1.703 sec)
INFO:tensorflow:global_step/sec: 62.389
INFO:tensorflow:loss = 7.56428, step = 8101 (1.601 sec)
INFO:tensorflow:global_step/sec: 69.6628
INFO:tensorflow:loss = 4.73977, step = 8201 (1.435 sec)
INFO:tensorflow:global_step/sec: 48.4644
INFO:tensorflow:loss = 7.85998, step = 8301 (2.065 sec)
INFO:tensorflow:global_step/sec: 62.0602
INFO:tensorflow:loss = 6.39131, step = 8401 (1.609 sec)
INFO:tensorflow:global_step/sec: 71.5862
INFO:tensorflow:loss = 14.1189, step = 8501 (1.397 sec)
INFO:tensorflow:global_step/sec: 52.8087
INFO:tensorflow:loss = 12.1342, step = 8601 (1.894 sec)
INFO:tensorflow:global_step/sec: 62.7249
INFO:tensorflow:loss = 4.8915, step = 8701 (1.596 sec)
INFO:tensorflow:global_step/sec: 62.2642
INFO:tensorflow:loss = 11.0851, step = 8801 (1.604 sec)
INFO:tensorflow:global_step/sec: 66.7522
INFO:tensorflow:loss = 4.0989, step = 8901 (1.498 sec)
INFO:tensorflow:global_step/sec: 66.1793
INFO:tensorflow:loss = 1.84748, step = 9001 (1.511 sec)
INFO:tensorflow:global_step/sec: 66.8259
INFO:tensorflow:loss = 3.20477, step = 9101 (1.496 sec)
INFO:tensorflow:global_step/sec: 59.0497
INFO:tensorflow:loss = 9.07242, step = 9201 (1.695 sec)
INFO:tensorflow:global_step/sec: 52.7599
INFO:tensorflow:loss = 2.82035, step = 9301 (1.894 sec)
INFO:tensorflow:global_step/sec: 61.8954
INFO:tensorflow:loss = 3.23482, step = 9401 (1.615 sec)
INFO:tensorflow:global_step/sec: 59.2671
INFO:tensorflow:loss = 2.7215, step = 9501 (1.688 sec)
INFO:tensorflow:global_step/sec: 62.0381
INFO:tensorflow:loss = 6.26391, step = 9601 (1.611 sec)
INFO:tensorflow:global_step/sec: 52.917
INFO:tensorflow:loss = 10.2159, step = 9701 (1.890 sec)
INFO:tensorflow:global_step/sec: 55.5259
INFO:tensorflow:loss = 9.42682, step = 9801 (1.801 sec)
INFO:tensorflow:global_step/sec: 66.5029
INFO:tensorflow:loss = 12.5752, step = 9901 (1.503 sec)
INFO:tensorflow:global_step/sec: 66.1434
INFO:tensorflow:loss = 10.2458, step = 10001 (1.512 sec)
INFO:tensorflow:global_step/sec: 62.9436
INFO:tensorflow:loss = 2.65545, step = 10101 (1.589 sec)
INFO:tensorflow:global_step/sec: 71.5901
INFO:tensorflow:loss = 2.37675, step = 10201 (1.397 sec)
INFO:tensorflow:global_step/sec: 66.2973
INFO:tensorflow:loss = 4.77555, step = 10301 (1.508 sec)
INFO:tensorflow:global_step/sec: 62.1294
INFO:tensorflow:loss = 8.18309, step = 10401 (1.609 sec)
INFO:tensorflow:global_step/sec: 59.6997
INFO:tensorflow:loss = 9.54808, step = 10501 (1.675 sec)
INFO:tensorflow:global_step/sec: 55.1247
INFO:tensorflow:loss = 1.80476, step = 10601 (1.814 sec)
INFO:tensorflow:global_step/sec: 66.7733
INFO:tensorflow:loss = 6.50839, step = 10701 (1.497 sec)
INFO:tensorflow:global_step/sec: 66.6505
INFO:tensorflow:loss = 4.24331, step = 10801 (1.500 sec)
INFO:tensorflow:global_step/sec: 55.879
INFO:tensorflow:loss = 13.358, step = 10901 (1.789 sec)
INFO:tensorflow:global_step/sec: 62.0916
INFO:tensorflow:loss = 7.45668, step = 11001 (1.611 sec)
INFO:tensorflow:global_step/sec: 52.9397
INFO:tensorflow:loss = 6.84107, step = 11101 (1.889 sec)
INFO:tensorflow:global_step/sec: 62.5447
INFO:tensorflow:loss = 4.74012, step = 11201 (1.599 sec)
INFO:tensorflow:global_step/sec: 65.9246
INFO:tensorflow:loss = 4.4362, step = 11301 (1.599 sec)
INFO:tensorflow:global_step/sec: 66.5267
INFO:tensorflow:loss = 1.79998, step = 11401 (1.500 sec)
INFO:tensorflow:global_step/sec: 59.206
INFO:tensorflow:loss = 12.167, step = 11501 (1.610 sec)
INFO:tensorflow:global_step/sec: 66.991
INFO:tensorflow:loss = 2.93986, step = 11601 (1.493 sec)
INFO:tensorflow:global_step/sec: 70.7423
INFO:tensorflow:loss = 6.62872, step = 11701 (1.413 sec)
INFO:tensorflow:global_step/sec: 59.3226
INFO:tensorflow:loss = 9.38785, step = 11801 (1.685 sec)
INFO:tensorflow:global_step/sec: 58.86
INFO:tensorflow:loss = 5.04938, step = 11901 (1.899 sec)
INFO:tensorflow:global_step/sec: 55.2269
INFO:tensorflow:loss = 8.66158, step = 12001 (1.611 sec)
INFO:tensorflow:global_step/sec: 66.5097
INFO:tensorflow:loss = 6.62202, step = 12101 (1.503 sec)
INFO:tensorflow:global_step/sec: 62.7002
INFO:tensorflow:loss = 6.12852, step = 12201 (1.595 sec)
INFO:tensorflow:global_step/sec: 62.593
INFO:tensorflow:loss = 3.37703, step = 12301 (1.598 sec)
INFO:tensorflow:global_step/sec: 58.6526
INFO:tensorflow:loss = 2.0968, step = 12401 (1.792 sec)
INFO:tensorflow:global_step/sec: 62.9158
INFO:tensorflow:loss = 7.35824, step = 12501 (1.502 sec)
INFO:tensorflow:global_step/sec: 71.021
INFO:tensorflow:loss = 4.83631, step = 12601 (1.408 sec)
INFO:tensorflow:global_step/sec: 58.9876
INFO:tensorflow:loss = 1.85133, step = 12701 (1.695 sec)
INFO:tensorflow:global_step/sec: 52.5123
INFO:tensorflow:loss = 2.11502, step = 12801 (1.904 sec)
INFO:tensorflow:global_step/sec: 62.7736
INFO:tensorflow:loss = 2.72864, step = 12901 (1.593 sec)
INFO:tensorflow:global_step/sec: 58.451
INFO:tensorflow:loss = 3.97841, step = 13001 (1.711 sec)
INFO:tensorflow:global_step/sec: 62.7049
INFO:tensorflow:loss = 11.2965, step = 13101 (1.595 sec)
INFO:tensorflow:global_step/sec: 58.7487
INFO:tensorflow:loss = 11.6658, step = 13201 (1.702 sec)
INFO:tensorflow:global_step/sec: 52.8437
INFO:tensorflow:loss = 4.72883, step = 13301 (1.892 sec)
INFO:tensorflow:global_step/sec: 52.5559
INFO:tensorflow:loss = 4.83573, step = 13401 (1.908 sec)
INFO:tensorflow:global_step/sec: 62.2096
INFO:tensorflow:loss = 8.99661, step = 13501 (1.692 sec)
INFO:tensorflow:global_step/sec: 58.879
INFO:tensorflow:loss = 1.46018, step = 13601 (1.609 sec)
INFO:tensorflow:global_step/sec: 62.8967
INFO:tensorflow:loss = 1.22616, step = 13701 (1.590 sec)
INFO:tensorflow:global_step/sec: 43.4568
INFO:tensorflow:loss = 1.85968, step = 13801 (2.302 sec)
INFO:tensorflow:global_step/sec: 66.7098
INFO:tensorflow:loss = 2.79163, step = 13901 (1.499 sec)
INFO:tensorflow:global_step/sec: 62.2024
INFO:tensorflow:loss = 8.51551, step = 14001 (1.608 sec)
INFO:tensorflow:global_step/sec: 62.6752
INFO:tensorflow:loss = 3.44269, step = 14101 (1.598 sec)
INFO:tensorflow:global_step/sec: 62.6373
INFO:tensorflow:loss = 5.5648, step = 14201 (1.594 sec)
INFO:tensorflow:global_step/sec: 58.2751
INFO:tensorflow:loss = 5.48194, step = 14301 (1.716 sec)
INFO:tensorflow:global_step/sec: 58.923
INFO:tensorflow:loss = 3.21667, step = 14401 (1.698 sec)
INFO:tensorflow:global_step/sec: 71.4846
INFO:tensorflow:loss = 3.45093, step = 14501 (1.401 sec)
INFO:tensorflow:global_step/sec: 71.4924
INFO:tensorflow:loss = 8.94037, step = 14601 (1.396 sec)
INFO:tensorflow:global_step/sec: 65.6546
INFO:tensorflow:loss = 4.23431, step = 14701 (1.523 sec)
INFO:tensorflow:global_step/sec: 56.3113
INFO:tensorflow:loss = 5.0881, step = 14801 (1.776 sec)
INFO:tensorflow:global_step/sec: 55.4434
INFO:tensorflow:loss = 7.08764, step = 14901 (1.888 sec)
INFO:tensorflow:global_step/sec: 62.3519
INFO:tensorflow:loss = 7.98236, step = 15001 (1.519 sec)
INFO:tensorflow:global_step/sec: 67.0226
INFO:tensorflow:loss = 3.29371, step = 15101 (1.492 sec)
INFO:tensorflow:global_step/sec: 62.7911
INFO:tensorflow:loss = 3.65796, step = 15201 (1.593 sec)
INFO:tensorflow:global_step/sec: 70.8896
INFO:tensorflow:loss = 6.20197, step = 15301 (1.410 sec)
INFO:tensorflow:global_step/sec: 62.6965
INFO:tensorflow:loss = 1.49016, step = 15401 (1.595 sec)
INFO:tensorflow:global_step/sec: 62.4318
INFO:tensorflow:loss = 1.65005, step = 15501 (1.601 sec)
INFO:tensorflow:global_step/sec: 62.8338
INFO:tensorflow:loss = 3.2754, step = 15601 (1.592 sec)
INFO:tensorflow:global_step/sec: 66.3715
INFO:tensorflow:loss = 1.44377, step = 15701 (1.507 sec)
INFO:tensorflow:global_step/sec: 62.1721
INFO:tensorflow:loss = 8.97553, step = 15801 (1.609 sec)
INFO:tensorflow:global_step/sec: 66.8652
INFO:tensorflow:loss = 7.00178, step = 15901 (1.496 sec)
INFO:tensorflow:global_step/sec: 67.1619
INFO:tensorflow:loss = 2.45056, step = 16001 (1.491 sec)
INFO:tensorflow:global_step/sec: 70.9643
INFO:tensorflow:loss = 2.03676, step = 16101 (1.409 sec)
INFO:tensorflow:global_step/sec: 67.0895
INFO:tensorflow:loss = 3.01222, step = 16201 (1.490 sec)
INFO:tensorflow:global_step/sec: 58.289
INFO:tensorflow:loss = 4.2512, step = 16301 (1.796 sec)
INFO:tensorflow:global_step/sec: 56.0495
INFO:tensorflow:loss = 7.4468, step = 16401 (1.702 sec)
INFO:tensorflow:global_step/sec: 62.455
INFO:tensorflow:loss = 7.53804, step = 16501 (1.601 sec)
INFO:tensorflow:global_step/sec: 58.7013
INFO:tensorflow:loss = 1.24997, step = 16601 (1.704 sec)
INFO:tensorflow:global_step/sec: 62.3069
INFO:tensorflow:loss = 3.44681, step = 16701 (1.605 sec)
INFO:tensorflow:global_step/sec: 62.3789
INFO:tensorflow:loss = 3.37909, step = 16801 (1.603 sec)
INFO:tensorflow:global_step/sec: 67.2195
INFO:tensorflow:loss = 3.15107, step = 16901 (1.487 sec)
INFO:tensorflow:global_step/sec: 66.47
INFO:tensorflow:loss = 1.51554, step = 17001 (1.505 sec)
INFO:tensorflow:global_step/sec: 66.3423
INFO:tensorflow:loss = 2.61759, step = 17101 (1.507 sec)
INFO:tensorflow:global_step/sec: 71.6806
INFO:tensorflow:loss = 7.31032, step = 17201 (1.395 sec)
INFO:tensorflow:global_step/sec: 58.5228
INFO:tensorflow:loss = 1.44817, step = 17301 (1.709 sec)
INFO:tensorflow:global_step/sec: 62.8317
INFO:tensorflow:loss = 6.4842, step = 17401 (1.591 sec)
INFO:tensorflow:global_step/sec: 70.9055
INFO:tensorflow:loss = 1.23821, step = 17501 (1.491 sec)
INFO:tensorflow:global_step/sec: 59.4282
INFO:tensorflow:loss = 1.71916, step = 17601 (1.602 sec)
INFO:tensorflow:global_step/sec: 62.3814
INFO:tensorflow:loss = 1.12171, step = 17701 (1.603 sec)
INFO:tensorflow:global_step/sec: 45.1835
INFO:tensorflow:loss = 6.56466, step = 17801 (2.213 sec)
INFO:tensorflow:global_step/sec: 59.3089
INFO:tensorflow:loss = 1.78898, step = 17901 (1.686 sec)
INFO:tensorflow:global_step/sec: 71.4822
INFO:tensorflow:loss = 3.41242, step = 18001 (1.399 sec)
INFO:tensorflow:global_step/sec: 62.5135
INFO:tensorflow:loss = 3.67466, step = 18101 (1.605 sec)
INFO:tensorflow:global_step/sec: 62.1726
INFO:tensorflow:loss = 6.63775, step = 18201 (1.604 sec)
INFO:tensorflow:global_step/sec: 66.2871
INFO:tensorflow:loss = 0.73981, step = 18301 (1.508 sec)
INFO:tensorflow:global_step/sec: 50.0412
INFO:tensorflow:loss = 6.49247, step = 18401 (2.000 sec)
INFO:tensorflow:global_step/sec: 66.9418
INFO:tensorflow:loss = 0.60717, step = 18501 (1.492 sec)
INFO:tensorflow:global_step/sec: 62.6991
INFO:tensorflow:loss = 1.11131, step = 18601 (1.595 sec)
INFO:tensorflow:global_step/sec: 62.3879
INFO:tensorflow:loss = 0.907614, step = 18701 (1.603 sec)
INFO:tensorflow:global_step/sec: 62.4047
INFO:tensorflow:loss = 2.1291, step = 18801 (1.603 sec)
INFO:tensorflow:global_step/sec: 70.7539
INFO:tensorflow:loss = 4.11129, step = 18901 (1.413 sec)
INFO:tensorflow:global_step/sec: 63.3963
INFO:tensorflow:loss = 5.44846, step = 19001 (1.577 sec)
INFO:tensorflow:global_step/sec: 49.9234
INFO:tensorflow:loss = 5.62491, step = 19101 (2.008 sec)
INFO:tensorflow:global_step/sec: 52.7318
INFO:tensorflow:loss = 7.74991, step = 19201 (1.893 sec)
INFO:tensorflow:global_step/sec: 66.4908
INFO:tensorflow:loss = 4.44105, step = 19301 (1.503 sec)
INFO:tensorflow:global_step/sec: 66.7745
INFO:tensorflow:loss = 2.3852, step = 19401 (1.500 sec)
INFO:tensorflow:global_step/sec: 62.3077
INFO:tensorflow:loss = 1.24064, step = 19501 (1.602 sec)
INFO:tensorflow:global_step/sec: 66.9137
INFO:tensorflow:loss = 3.32384, step = 19601 (1.494 sec)
INFO:tensorflow:global_step/sec: 66.5352
INFO:tensorflow:loss = 1.18686, step = 19701 (1.503 sec)
INFO:tensorflow:global_step/sec: 71.2248
INFO:tensorflow:loss = 1.4706, step = 19801 (1.404 sec)
INFO:tensorflow:global_step/sec: 58.7617
INFO:tensorflow:loss = 2.12143, step = 19901 (1.702 sec)
INFO:tensorflow:global_step/sec: 52.2433
INFO:tensorflow:loss = 5.88507, step = 20001 (1.914 sec)
INFO:tensorflow:global_step/sec: 67.2775
INFO:tensorflow:loss = 0.728642, step = 20101 (1.487 sec)
INFO:tensorflow:global_step/sec: 62.5287
INFO:tensorflow:loss = 4.13613, step = 20201 (1.599 sec)
INFO:tensorflow:global_step/sec: 62.2724
INFO:tensorflow:loss = 6.14378, step = 20301 (1.606 sec)
INFO:tensorflow:global_step/sec: 59.1515
INFO:tensorflow:loss = 1.61584, step = 20401 (1.690 sec)
INFO:tensorflow:global_step/sec: 70.4658
INFO:tensorflow:loss = 2.9576, step = 20501 (1.419 sec)
INFO:tensorflow:global_step/sec: 67.0172
INFO:tensorflow:loss = 4.18499, step = 20601 (1.573 sec)
INFO:tensorflow:global_step/sec: 58.8131
INFO:tensorflow:loss = 2.83805, step = 20701 (1.620 sec)
INFO:tensorflow:global_step/sec: 67.0521
INFO:tensorflow:loss = 5.63633, step = 20801 (1.491 sec)
INFO:tensorflow:global_step/sec: 66.6048
INFO:tensorflow:loss = 3.61582, step = 20901 (1.501 sec)
INFO:tensorflow:global_step/sec: 66.5481
INFO:tensorflow:loss = 2.29087, step = 21001 (1.502 sec)
INFO:tensorflow:global_step/sec: 62.6564
INFO:tensorflow:loss = 2.8032, step = 21101 (1.596 sec)
INFO:tensorflow:global_step/sec: 66.6184
INFO:tensorflow:loss = 1.69553, step = 21201 (1.501 sec)
INFO:tensorflow:global_step/sec: 71.6782
INFO:tensorflow:loss = 2.76058, step = 21301 (1.395 sec)
INFO:tensorflow:global_step/sec: 71.3994
INFO:tensorflow:loss = 3.61501, step = 21401 (1.401 sec)
INFO:tensorflow:global_step/sec: 52.4695
INFO:tensorflow:loss = 2.40454, step = 21501 (1.906 sec)
INFO:tensorflow:global_step/sec: 67.0995
INFO:tensorflow:loss = 4.98355, step = 21601 (1.491 sec)
INFO:tensorflow:global_step/sec: 66.3162
INFO:tensorflow:loss = 13.1601, step = 21701 (1.508 sec)
INFO:tensorflow:global_step/sec: 66.8712
INFO:tensorflow:loss = 3.45241, step = 21801 (1.495 sec)
INFO:tensorflow:global_step/sec: 66.8277
INFO:tensorflow:loss = 5.79841, step = 21901 (1.496 sec)
INFO:tensorflow:global_step/sec: 62.1822
INFO:tensorflow:loss = 1.91356, step = 22001 (1.609 sec)
INFO:tensorflow:global_step/sec: 59.1608
INFO:tensorflow:loss = 1.22457, step = 22101 (1.691 sec)
INFO:tensorflow:global_step/sec: 62.4456
INFO:tensorflow:loss = 2.17203, step = 22201 (1.601 sec)
INFO:tensorflow:global_step/sec: 66.1798
INFO:tensorflow:loss = 2.53251, step = 22301 (1.511 sec)
INFO:tensorflow:global_step/sec: 66.6269
INFO:tensorflow:loss = 6.91364, step = 22401 (1.501 sec)
INFO:tensorflow:global_step/sec: 59.1021
INFO:tensorflow:loss = 2.07916, step = 22501 (1.692 sec)
INFO:tensorflow:global_step/sec: 62.6792
INFO:tensorflow:loss = 3.86921, step = 22601 (1.596 sec)
INFO:tensorflow:global_step/sec: 55.3434
INFO:tensorflow:loss = 5.96153, step = 22701 (1.807 sec)
INFO:tensorflow:global_step/sec: 66.7819
INFO:tensorflow:loss = 5.21667, step = 22801 (1.497 sec)
INFO:tensorflow:global_step/sec: 62.7447
INFO:tensorflow:loss = 2.17067, step = 22901 (1.597 sec)
INFO:tensorflow:global_step/sec: 52.3343
INFO:tensorflow:loss = 4.34011, step = 23001 (1.908 sec)
INFO:tensorflow:global_step/sec: 66.8098
INFO:tensorflow:loss = 4.96399, step = 23101 (1.497 sec)
INFO:tensorflow:global_step/sec: 62.7562
INFO:tensorflow:loss = 4.86404, step = 23201 (1.593 sec)
INFO:tensorflow:global_step/sec: 66.0625
INFO:tensorflow:loss = 1.67714, step = 23301 (1.597 sec)
INFO:tensorflow:global_step/sec: 59.2796
INFO:tensorflow:loss = 4.59056, step = 23401 (1.604 sec)
INFO:tensorflow:global_step/sec: 66.0573
INFO:tensorflow:loss = 2.6657, step = 23501 (1.514 sec)
INFO:tensorflow:global_step/sec: 71.1182
INFO:tensorflow:loss = 5.5271, step = 23601 (1.406 sec)
INFO:tensorflow:global_step/sec: 56.2021
INFO:tensorflow:loss = 3.77428, step = 23701 (1.779 sec)
INFO:tensorflow:global_step/sec: 66.7144
INFO:tensorflow:loss = 5.46438, step = 23801 (1.499 sec)
INFO:tensorflow:global_step/sec: 70.6711
INFO:tensorflow:loss = 8.34928, step = 23901 (1.417 sec)
INFO:tensorflow:global_step/sec: 55.9896
INFO:tensorflow:loss = 6.51198, step = 24001 (1.785 sec)
INFO:tensorflow:global_step/sec: 66.24
INFO:tensorflow:loss = 6.46996, step = 24101 (1.508 sec)
INFO:tensorflow:global_step/sec: 66.412
INFO:tensorflow:loss = 4.98276, step = 24201 (1.510 sec)
INFO:tensorflow:global_step/sec: 66.939
INFO:tensorflow:loss = 5.06682, step = 24301 (1.490 sec)
INFO:tensorflow:global_step/sec: 66.5723
INFO:tensorflow:loss = 0.884624, step = 24401 (1.502 sec)
INFO:tensorflow:global_step/sec: 66.3561
INFO:tensorflow:loss = 3.55032, step = 24501 (1.507 sec)
INFO:tensorflow:global_step/sec: 62.322
INFO:tensorflow:loss = 1.41036, step = 24601 (1.614 sec)
INFO:tensorflow:global_step/sec: 63.2661
INFO:tensorflow:loss = 3.69165, step = 24701 (1.571 sec)
INFO:tensorflow:global_step/sec: 70.9835
INFO:tensorflow:loss = 0.625495, step = 24801 (1.409 sec)
INFO:tensorflow:global_step/sec: 62.4408
INFO:tensorflow:loss = 2.14022, step = 24901 (1.602 sec)
INFO:tensorflow:global_step/sec: 66.8896
INFO:tensorflow:loss = 1.82245, step = 25001 (1.495 sec)
INFO:tensorflow:global_step/sec: 62.736
INFO:tensorflow:loss = 1.94663, step = 25101 (1.594 sec)
INFO:tensorflow:global_step/sec: 70.1815
INFO:tensorflow:loss = 3.87769, step = 25201 (1.427 sec)
INFO:tensorflow:global_step/sec: 62.9132
INFO:tensorflow:loss = 5.28026, step = 25301 (1.587 sec)
INFO:tensorflow:global_step/sec: 71.6041
INFO:tensorflow:loss = 1.80573, step = 25401 (1.482 sec)
INFO:tensorflow:global_step/sec: 55.583
INFO:tensorflow:loss = 1.6548, step = 25501 (1.714 sec)
INFO:tensorflow:global_step/sec: 62.5768
INFO:tensorflow:loss = 4.10314, step = 25601 (1.599 sec)
INFO:tensorflow:global_step/sec: 66.6209
INFO:tensorflow:loss = 2.42826, step = 25701 (1.501 sec)
INFO:tensorflow:global_step/sec: 62.7834
INFO:tensorflow:loss = 4.4398, step = 25801 (1.593 sec)
INFO:tensorflow:global_step/sec: 62.3267
INFO:tensorflow:loss = 3.41601, step = 25901 (1.604 sec)
INFO:tensorflow:global_step/sec: 66.7089
INFO:tensorflow:loss = 2.13032, step = 26001 (1.499 sec)
INFO:tensorflow:global_step/sec: 50.229
INFO:tensorflow:loss = 2.95025, step = 26101 (1.991 sec)
INFO:tensorflow:global_step/sec: 66.4419
INFO:tensorflow:loss = 1.38499, step = 26201 (1.505 sec)
INFO:tensorflow:global_step/sec: 70.7909
INFO:tensorflow:loss = 7.4967, step = 26301 (1.413 sec)
INFO:tensorflow:global_step/sec: 67.1525
INFO:tensorflow:loss = 1.75697, step = 26401 (1.489 sec)
INFO:tensorflow:global_step/sec: 52.3003
INFO:tensorflow:loss = 0.849341, step = 26501 (1.912 sec)
INFO:tensorflow:global_step/sec: 56.1507
INFO:tensorflow:loss = 6.49441, step = 26601 (1.781 sec)
INFO:tensorflow:global_step/sec: 82.1044
INFO:tensorflow:loss = 1.43898, step = 26701 (1.299 sec)
INFO:tensorflow:global_step/sec: 62.9056
INFO:tensorflow:loss = 2.32675, step = 26801 (1.508 sec)
INFO:tensorflow:global_step/sec: 62.2052
INFO:tensorflow:loss = 0.363385, step = 26901 (1.608 sec)
INFO:tensorflow:global_step/sec: 62.7751
INFO:tensorflow:loss = 5.17893, step = 27001 (1.593 sec)
INFO:tensorflow:global_step/sec: 62.2846
INFO:tensorflow:loss = 3.20398, step = 27101 (1.606 sec)
INFO:tensorflow:global_step/sec: 62.646
INFO:tensorflow:loss = 2.44348, step = 27201 (1.596 sec)
INFO:tensorflow:global_step/sec: 62.7066
INFO:tensorflow:loss = 4.14998, step = 27301 (1.594 sec)
INFO:tensorflow:global_step/sec: 66.2015
INFO:tensorflow:loss = 1.43873, step = 27401 (1.511 sec)
INFO:tensorflow:global_step/sec: 66.6004
INFO:tensorflow:loss = 4.35953, step = 27501 (1.501 sec)
INFO:tensorflow:global_step/sec: 67.1299
INFO:tensorflow:loss = 4.98034, step = 27601 (1.489 sec)
INFO:tensorflow:global_step/sec: 66.8949
INFO:tensorflow:loss = 0.457042, step = 27701 (1.496 sec)
INFO:tensorflow:global_step/sec: 62.5492
INFO:tensorflow:loss = 2.2295, step = 27801 (1.601 sec)
INFO:tensorflow:global_step/sec: 62.119
INFO:tensorflow:loss = 1.33891, step = 27901 (1.607 sec)
INFO:tensorflow:global_step/sec: 55.5105
INFO:tensorflow:loss = 0.864555, step = 28001 (1.802 sec)
INFO:tensorflow:global_step/sec: 58.6545
INFO:tensorflow:loss = 0.84185, step = 28101 (1.788 sec)
INFO:tensorflow:global_step/sec: 55.6414
INFO:tensorflow:loss = 1.14384, step = 28201 (1.717 sec)
INFO:tensorflow:global_step/sec: 62.9771
INFO:tensorflow:loss = 3.7685, step = 28301 (1.585 sec)
INFO:tensorflow:global_step/sec: 70.7512
INFO:tensorflow:loss = 2.43438, step = 28401 (1.414 sec)
INFO:tensorflow:global_step/sec: 63.0453
INFO:tensorflow:loss = 1.55619, step = 28501 (1.586 sec)
INFO:tensorflow:global_step/sec: 66.075
INFO:tensorflow:loss = 1.6467, step = 28601 (1.513 sec)
INFO:tensorflow:global_step/sec: 55.6513
INFO:tensorflow:loss = 1.44003, step = 28701 (1.797 sec)
INFO:tensorflow:global_step/sec: 58.6934
INFO:tensorflow:loss = 0.898293, step = 28801 (1.704 sec)
INFO:tensorflow:global_step/sec: 52.9944
INFO:tensorflow:loss = 0.452232, step = 28901 (1.887 sec)
INFO:tensorflow:global_step/sec: 62.3262
INFO:tensorflow:loss = 0.909782, step = 29001 (1.604 sec)
INFO:tensorflow:global_step/sec: 66.6463
INFO:tensorflow:loss = 4.25549, step = 29101 (1.501 sec)
INFO:tensorflow:global_step/sec: 62.6922
INFO:tensorflow:loss = 1.46972, step = 29201 (1.598 sec)
INFO:tensorflow:global_step/sec: 66.5259
INFO:tensorflow:loss = 0.90285, step = 29301 (1.502 sec)
INFO:tensorflow:global_step/sec: 70.934
INFO:tensorflow:loss = 1.59605, step = 29401 (1.493 sec)
INFO:tensorflow:global_step/sec: 55.5452
INFO:tensorflow:loss = 2.24467, step = 29501 (1.718 sec)
INFO:tensorflow:global_step/sec: 62.767
INFO:tensorflow:loss = 2.64073, step = 29601 (1.591 sec)
INFO:tensorflow:global_step/sec: 58.6681
INFO:tensorflow:loss = 1.61933, step = 29701 (1.708 sec)
INFO:tensorflow:global_step/sec: 62.565
INFO:tensorflow:loss = 3.28505, step = 29801 (1.595 sec)
INFO:tensorflow:global_step/sec: 62.9197
INFO:tensorflow:loss = 0.954338, step = 29901 (1.592 sec)
INFO:tensorflow:Saving checkpoints for 30000 into ./tmp/mnist_model/model.ckpt.
INFO:tensorflow:Loss for final step: 2.16675.

```





```
<tensorflow.python.estimator.canned.dnn.DNNClassifier at 0x7f780c5a71d0>

```



In order to estimate the accuracy of the model, we need to define another "input function"

```python
# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_test},
    y=y_test.astype(np.int32),
    num_epochs=1,
    shuffle=False
)


```

```python
# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

```

```
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-01-15-01:08:47
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./tmp/mnist_model/model.ckpt-30000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-01-15-01:08:50
INFO:tensorflow:Saving dict for global step 30000: accuracy = 0.9809, average_loss = 0.0659267, global_step = 30000, loss = 8.34515
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 30000: ./tmp/mnist_model/model.ckpt-30000

Test Accuracy: 98.089999%

```



I got accuracy about 98.089999% on eval set.

# <a name="section4"></a> Create TensorFlow model using TensorFlow's Estimator API

tf.estimator

In previous sections, we learned how to **Kereas** and **estimator** API to build a simple model. The dataset is downloaded and saved in numpy arrays. When you need to hundle a large dataset like 10GB. The memory can handle it in pandas or numpy. TF API need to read data directly from csv file lines batch by batch. They are good for distributed learning as well.  
This section is modifed from the google cloud platform repository [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/06_structured/3_tensorflow_dnn.ipynb). 

**add and save files for training and testing.csv**

```python
pd.DataFrame(x_train.reshape(-1,dim*dim)).join(pd.DataFrame(y_train,columns=['label'])).to_csv('train.csv',index=False,header=False)

```

I seem it will take about 1 minute to compile those files and save in pandas.

```python
pd.DataFrame(x_test.reshape(-1,dim*dim)).join(pd.DataFrame(y_test,columns=['label'])).to_csv('test.csv',index=False,header=False)

```

let us see how the df look like

```python
temp=pd.read_csv('train.csv',header=None)

```

```python
temp.head()

```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}

```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>775</th>
      <th>776</th>
      <th>777</th>
      <th>778</th>
      <th>779</th>
      <th>780</th>
      <th>781</th>
      <th>782</th>
      <th>783</th>
      <th>784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>5 rows Ã— 785 columns</p>

</div>



```python
temp2=pd.read_csv('test.csv',header=None)
temp2.head()

```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```

temp=pd.read_csv('train.csv',header=None)

temp.head()
}

```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>775</th>
      <th>776</th>
      <th>777</th>
      <th>778</th>
      <th>779</th>
      <th>780</th>
      <th>781</th>
      <th>782</th>
      <th>783</th>
      <th>784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows Ã— 785 columns</p>

</div>



Columns 0-783 are pixels for the image; and column 784 is the label

#### Define read_dataset

We have to create a input_function to read data from files batch by batch and pack them into a dict with "key" equal to column names and "value" equal to values as input data source.

```python
#define columns here
CSV_COLUMNS = ["pixel"+str(i) for i in range (dim*dim)]+['digit']
LABEL_COLUMN = 'digit'
DEFAULTS = [[0.0] for i in range(dim*dim)]+[[0]] # [[0],['NA'],[0]]

def read_dataset(filename, mode, batch_size=32):
    def _input_fn():
        def decode_csv(value_colum):
            columns = tf.decode_csv(value_colum,record_defaults = DEFAULTS)
            features= dict(zip(CSV_COLUMNS,columns))
            label = features.pop (LABEL_COLUMN)
            return features, label
        
        # Create list of files that match pattern
        file_list =  tf.gfile.Glob(filename)
        
        # Create dataset from file list
        dataset = tf.data.TextLineDataset(filename).map(decode_csv) # Transform each elem by applying decode_csv fn
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None   # indefinitely
            dataset = dataset.shuffle(buffer_size =10 *batch_size)
        else:
            num_epochs = 1   # end-of-input after this
        dataset = dataset.repeat(num_epochs).batch(batch_size)

        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

```

#### Define a feature columns

Because we have grey scale values for the pixels we can just use **numeric_column** for those columns.

```python
# Define feature columns  - not including label
def get_cols():
  # Define column types
  return [tf.feature_column.numeric_column('pixel'+str(i)) for i in range(dim*dim)]


```



```
''

```



#### Define a severing function

by defining the severing input, we could use it to evaluate the model. The serving functing is being used as **exporters** in **EvalSpec**. Basically, it tell what is the format when evaluate the model, it should be same as the traning format. Export your model to work with JSON dictionaries.

```python
# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    csv_row = tf.placeholder(shape=[None], dtype=tf.string)
    columns = tf.decode_csv(csv_row,record_defaults = DEFAULTS[:-1])
    features= dict(zip(CSV_COLUMNS,columns))
    #feature_placeholders = dict(zip(['pixel'+str(i) for i in range(dim*dim)],  
    #                                [tf.placeholder(tf.float32, [None]) for i in range(dim*dim)]))
    
    #features = {key: tf.expand_dims(tensor,-1) for key,tensor in feature_placeholders.items()}

    return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})#feature_placeholders)

```

This function we define a **train_and_evaluate** function to complile all parts together.  
It include three main section: *estimator*, *train_spec*, and *eval_spec*. 
*estimator* : here we just use a simple DNNClassifier here, we can also build Convolutional models as well later.
*train_spec* : pass input_fn as we defined before, and traning steps.
*eval_spec*: define the evaluation freqencies

```python
def train_and_evaluate(output_dir):
    EVAL_INTERVAL = 300  #save checkpoint every 300s
    TRAIN_STEPS = 300
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL)
    
    estimator = tf.estimator.DNNClassifier (model_dir=output_dir, 
                                         feature_columns = get_cols(),hidden_units=[32],
                                            dropout=0.2,
                                            n_classes=10,
                                        config = run_config)

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    train_spec = tf.estimator.TrainSpec(input_fn =  read_dataset('train.csv', mode = tf.contrib.learn.ModeKeys.TRAIN),
                                        max_steps= TRAIN_STEPS)

    
    eval_spec = tf.estimator.EvalSpec(input_fn= read_dataset('test.csv', mode = tf.contrib.learn.ModeKeys.EVAL), 
                                        steps = None,
                                        start_delay_secs = 60, # start evaluating after N seconds
                                        throttle_secs= EVAL_INTERVAL,  # evaluate every N seconds
                                        exporters = exporter)

    tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)

```

```python
import shutil
shutil.rmtree('mnist', ignore_errors = True)
train_and_evaluate('mnist')

```

```
INFO:tensorflow:Using config: {'_save_summary_steps': 100, '_save_checkpoints_steps': None, '_global_id_in_cluster': 0, '_task_id': 0, '_task_type': 'worker', '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f780c2ae198>, '_num_worker_replicas': 1, '_protocol': None, '_eval_distribute': None, '_master': '', '_is_chief': True, '_tf_random_seed': None, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_num_ps_replicas': 0, '_keep_checkpoint_every_n_hours': 10000, '_save_checkpoints_secs': 300, '_train_distribute': None, '_experimental_distribute': None, '_model_dir': 'mnist', '_log_step_count_steps': 100, '_device_fn': None, '_keep_checkpoint_max': 5, '_service': None, '_evaluation_master': ''}
INFO:tensorflow:Not using Distribute Coordinator.
INFO:tensorflow:Running training and evaluation locally (non-distributed).
INFO:tensorflow:Start train and evaluate loop. The evaluate will happen after every checkpoint. Checkpoint frequency is determined based on RunConfig arguments: save_checkpoints_steps None or save_checkpoints_secs 300.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into mnist/model.ckpt.
INFO:tensorflow:loss = 75.8332, step = 1
INFO:tensorflow:global_step/sec: 8.96992
INFO:tensorflow:loss = 7.4517, step = 101 (11.153 sec)
INFO:tensorflow:global_step/sec: 12.7129
INFO:tensorflow:loss = 11.7008, step = 201 (7.863 sec)
INFO:tensorflow:Saving checkpoints for 300 into mnist/model.ckpt.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-01-15-02:18:11
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from mnist/model.ckpt-300
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-01-15-02:18:37
INFO:tensorflow:Saving dict for global step 300: accuracy = 0.9088, average_loss = 0.335611, global_step = 300, loss = 10.7224
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 300: mnist/model.ckpt-300
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Signatures INCLUDED in export for Train: None
INFO:tensorflow:Signatures INCLUDED in export for Regress: None
INFO:tensorflow:Signatures INCLUDED in export for Eval: None
INFO:tensorflow:Signatures INCLUDED in export for Classify: ['classification', 'serving_default']
INFO:tensorflow:Signatures INCLUDED in export for Predict: ['predict']
INFO:tensorflow:Restoring parameters from mnist/model.ckpt-300
INFO:tensorflow:Assets added to graph.
INFO:tensorflow:No assets to write.
INFO:tensorflow:SavedModel written to: mnist/export/exporter/temp-b'1547518717'/saved_model.pb
INFO:tensorflow:Loss for final step: 8.53443.

```

