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
...
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
