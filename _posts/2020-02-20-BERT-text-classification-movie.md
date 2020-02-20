---
title: BERT-text-classification-movie
subtitle: 
published: false
date: '' 
tags:
    - jupyter
    - python
    - notebook
layout: post

---
# BERT text classification on movie dataset

In this notebook, we will use Hugging face
[Transformers](https://huggingface.co/transformers/) to build BERT model on text
classification task with [Tensorflow
2.0](https://www.tensorflow.org/guide/effective_tf2).

Notes: this notebook is entirely run on [Google
colab](https://colab.research.google.com/) with GPU. If you start a new
notebook, you need to choose "Runtime"->"Change runtime type" ->"GPU" at the
begining. You can also find in [this notebook](https://drive.google.com/open?id=1kEg0SnYNtw_IJwu_kl5y3qRVs-BKBmNO).

### Introduction of BERT

[BERT](https://https://arxiv.org/abs/1810.04805)  ( which stands for
Bidirectional Encoder Representations from
Transformers) as a languge model was introduced by Jacob et. al. at 2018 from
Google. BERT is designed to pretrain deep bidirectional representations from
unlabeled text by jointly conditioning on both left and right context in all
layers.

The core of BERT is tranformer. For those who don't know what is transoformer,
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) and this
[blog](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-
is-all-you-need/#.Xk1qTChKhPb) are good resources to get an ideal how attention
mechanism works and how transformer works.



### Introduction of huggingface or Transformers

Hugging face is a company which invented a pacakge called
[Transformers](https://github.com/huggingface/transformers). It provides state-
of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert,
XLNet, CTRL...) for Natural Language Understanding (NLU) and Natural Language
Generation (NLG) with over 32+ pretrained models in 100+ languages and deep
interoperability between TensorFlow 2.0 and PyTorch.

TensorFlow uses layers of abstraction when putting together a model. Operations
(such as matrix algebra) can occur at a low level, and an abstraction of a
neural network layer can occur at a higher level. One of these higher levels of
abstraction is called a Keras model, and Huggingface uses this model as a way to
abstract some of the granular details involved in using BERT for transfer
learning.

The Transformer is mainly developed based on the pytorch but the TensorFLow 2.0
version implementation on BERT is also fantasic. I am a fan of tensorflow, so in
this notebook, we will implement a classification task using Transformers with
TF 2.0.

# Run BERT

Before we will start taking about the model and data, we need setup packages in
Google colab.
So we need to upgrade **pip**, install **tensorflow-gpu 2.0** and
**transformers**. It takes about 1 min to run. You may notice here, I use
tensorflow-gpu instead of tensorflow, becuase bert use a lot of computation
resources, gpu will run much faster than cpu. The good thing is, you only need
the gpu package installed. All other operations are same in regular tensorflow.
The TF infrastructure will handle it.

## Install package

{% highlight python %}
!pip install -q --upgrade pip
!pip install -q tensorflow-gpu==2.0.0
!pip install -q transformers
{% endhighlight %}


## Prepare dataset

### Download data

In this example , we will use standford movie review sentiment analysis data, it
has been upload on [Kaggle](https://www.kaggle.com/atulanandjha/imdb-50k-movie-
reviews-test-your-bert). It contian 25000 train and 25000 test review texts and
labels as 'pos' and 'neg', which has been anotaited their sentiments. Task is to
use text to build claissifier to identify it's sentiment.  I have downloaded it
and upload into my repo.


{% highlight python %}
!git clone https://github.com/yuyongze/movie-sst2.git
%cd movie-sst2
{% endhighlight %}

    Cloning into 'movie-sst2'...
    remote: Enumerating objects: 8, done.[K
    remote: Counting objects: 100% (8/8), done.[K
    remote: Compressing objects: 100% (7/7), done.[K
    remote: Total 8 (delta 0), reused 8 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (8/8), done.
    /content/movie-sst2


### Load data


{% highlight python %}
import tensorflow as tf
import pandas as pd
from tensorflow.python.lib.io.tf_record import TFRecordWriter
{% endhighlight %}

Read csv file, we notice that the sentiment is labeled as 'pos' and 'neg', we
replace the 'pos' as 1 and 'neg' as 0,
And inthis example, we are using only 20% data (**5000 examples**) as a toy
example to demonstrate the model can work well.


{% highlight python %}
# fraction of sample pass to the train and test as example
SAMPLE_FRAC = 0.2
# 80% data for training and 20% data for validate
TRAIN_FRAC = 0.8
# load train data from train.csv
train = pd.read_csv('./data/train.csv')
train.reset_index(inplace=True)
# change sentiment label form 'pos' and 'neg' to 1 and 0, which bert model knows
train['sentiment'].replace({'pos':1,'neg':0},inplace=True)

# train set
train_sample = train.sample(frac=SAMPLE_FRAC,random_state=0)
train_select = train_sample.sample(frac= TRAIN_FRAC,random_state=0)
train_csv = train_select.values

# validate set 
validate_select = train_sample.drop(index=train_select.index)
validate_csv = validate_select.values


# load test data , here should be validation set
test = pd.read_csv('./data/test.csv')
test.reset_index(inplace=True)
test['sentiment'].replace({'pos':1,'neg':0},inplace=True)
test_csv = test.sample(frac=SAMPLE_FRAC,random_state=0).values
{% endhighlight %}


{% highlight python %}
train.tail()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>text</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24995</th>
      <td>24995</td>
      <td>This film is fun, if your a person who likes a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>24996</td>
      <td>After seeing this film I feel like I know just...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>24997</td>
      <td>first this deserves about 5 stars due to actin...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>24998</td>
      <td>If you like films that ramble with little plot...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>24999</td>
      <td>As interesting as a sheet of cardboard, this d...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



As you can see the *train_csv*,*validate_csv*, and *test_csv* has 3 columns,
which are 'index','text',and 'sentiment'. They are important, becuase we need to
pack those three parts into examples and feed to the models.

### Build TFRecord

We have training data and validate data ready, and now we need convert those
data into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)
which tensorflow can read it into tf.data.Dataset object easily. In this
[guide](https://www.tensorflow.org/guide/data), you can find how to transfer
data into `tf.data.Dataset`.


{% highlight python %}
import time
def create_tf_example(features,label):
    """
    Create tf example using features and label

    Args:
        features: list, feature list with format  ['idx','sentence']
        label: string, 
    
    Return:
        A binary-string of tf example.
        All proto messages can be serialized to a binary-string using the .SerializeToString method.
    """
    tf_example = tf.train.Example(features = tf.train.Features(feature = {
        'idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[0]])),
        'sentence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[1].encode('utf-8')])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))
    return tf_example.SerializeToString()

def convert_csv_to_tfrecord(csv, file_name):
    """
    Convert the numpy arryes to tfrecord and write files

    Args:
        csv: numpy arrays, each row feed (features+label)
        file_name: location TFRecord to be saved 
    """
    start_time = time.time()
    writer = TFRecordWriter(file_name)
    for idx,row in enumerate(csv):
        # check the row retionality, raise error when missing value
        try:
            if row is None:
                raise Exception('Row Missing')
            if row[0] is None or row[1] is None or row[2] is None:
                raise Exception('Value Missing')
            if row[1].strip() is '':
                raise Exception('Utterance is empty')
            
            features, label = row[:-1],row[-1]
            example =  create_tf_example(features,label)
            writer.write(example)
    
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
    writer.close()
    print(f"{file_name}: --- {(time.time() - start_time)} seconds ---")
{% endhighlight %}

convert csv(numpy files) to tfrecord using `convert_csv_to_tfrecord`


{% highlight python %}
convert_csv_to_tfrecord(train_csv, "./data/movie_train.tfrecord")
convert_csv_to_tfrecord(validate_csv, "./data/movie_validate.tfrecord")
convert_csv_to_tfrecord(test_csv, "./data/movie_test.tfrecord")
{% endhighlight %}

    ./data/movie_train.tfrecord: --- 0.16509032249450684 seconds ---
    data/movie_validate.tfrecord: --- 0.04649686813354492 seconds ---
    data/movie_test.tfrecord: --- 0.19834470748901367 seconds ---


now we will generate a json file to save number of training example to determine
the tranin steps in the later process


{% highlight python %}
import json
# generate exmaple number , save for use in the future 
def generate_json_info(local_file_name,df_train=[],df_val=[],df_test=[]):
    info = {"train_length": len(df_train), "validation_length": len(df_val),
            "test_length": len(df_test)}

    with open(local_file_name, 'w') as outfile:
        json.dump(info, outfile)

generate_json_info('./data/info.json',train_csv,validate_csv,test_csv)
{% endhighlight %}

#### Confirm that TFRecord has encoded correctly


{% highlight python %}
tr_ds = tf.data.TFRecordDataset("data/movie_train.tfrecord")
{% endhighlight %}

About how to write and read tensorflow TFRecord, you can read documentation
[here](https://www.tensorflow.org/tutorials/load_data/tfrecord).


{% highlight python %}
# Create a description of the features.
feature_spec = {
    'idx': tf.io.FixedLenFeature([], tf.int64),
    'sentence': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}
def parse_example(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_spec)
tr_parse_ds = tr_ds.map(parse_example)
dataset_iterator = iter(tr_parse_ds)
{% endhighlight %}


{% highlight python %}
dataset_iterator.get_next()
{% endhighlight %}




    {'idx': <tf.Tensor: id=259301, shape=(), dtype=int64, numpy=8383>,
     'label': <tf.Tensor: id=259302, shape=(), dtype=int64, numpy=1>,
     'sentence': <tf.Tensor: id=259303, shape=(), dtype=string, numpy=b"This movie is difficult to watch in our fast-paced culture of the 21st century, but it is worth it for the messages that it conveys, chiefly the consequences and ramifications of technology upon society, specifically when that technology is used for warfare.<br /><br />This movie presents a full circle cycle of dehumanization and rehumanization as influenced by the advent of technology and the subsequent deconstruction of civilization and therefore serves as a cautionary tale against the misuse of technology, but as the circle completes itself, familiar themes and sentiments pop up again to present self-serving rather than self-destructive ways that humanity may utilize technology.<br /><br />Brilliant for it's time, the picture and sound quality may pose a challenge for some, but as a landmark in the history, development, and evolution of the sci-fi genre, it is a must. In the end, free will and free choice are once again posed to humanity as a means for controlling our own destiny rather than having it served to us by someone else or indeed, the state of<br /><br />society itself, as shaped by world events.<br /><br />Those who are downtrodden by what life throws their way sometimes tend to remain so, but yet there is always a glimmer of hope and continuity that remains, as this film posits.<br /><br />As far as qualifying as sci-fi, one of the biggest common demoninators of that genre is it's speculative nature. It asks us the questions, what if these events happened this way, and what effect would it have on society or the individuals within it? How would we react?<br /><br />As far as influence, this film projects those speculative sciences that make sci-fi as unique as it is and keeps us asking those important questions.">}



## BERT Sentiment Classification in TensorFlow 2.0


{% highlight python %}
import tensorflow as tf
from transformers import *
from transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features
from transformers.configuration_bert import BertConfig
{% endhighlight %}

Load the TFRecord


{% highlight python %}
tr_ds = tf.data.TFRecordDataset("data/movie_train.tfrecord")
val_ds = tf.data.TFRecordDataset("data/movie_validate.tfrecord")
test_ds = tf.data.TFRecordDataset("data/movie_test.tfrecord")

{% endhighlight %}


{% highlight python %}
# Create a description of the features.
feature_spec = {
    'idx': tf.io.FixedLenFeature([], tf.int64),
    'sentence': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}
def parse_example(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_spec)

# convert the encoded string tensor into the separate tensors that will feed into the model
tr_parse_ds = tr_ds.map(parse_example)
val_parse_ds = val_ds.map(parse_example)
test_parse_ds =  test_ds.map(parse_example)
{% endhighlight %}

One approach to cleaning up a pipeline is to map a function to the dataset. In
this way, the function gets applied to each example. The following code uses
this approach to clean up the sentence tensor.




{% highlight python %}
def clean_string(features):
    revised_sentence = tf.strings.regex_replace(features['sentence'], "\.\.\.", "", replace_global=True)
    revised_sentence = tf.strings.regex_replace(revised_sentence, "\\'", "'", replace_global=True)
    revised_sentence = tf.strings.regex_replace(revised_sentence, "\\n", "", replace_global=True)
    features['sentence'] = revised_sentence
    return features
{% endhighlight %}


{% highlight python %}
tr_clean_ds = tr_parse_ds.map(lambda features: clean_string(features))
val_clean_ds = val_parse_ds.map(lambda features: clean_string(features))
test_clean_ds =  test_parse_ds.map(lambda features: clean_string(features))
{% endhighlight %}

### Train

Before training, we need to set up some paramerter ahead. `BATCH_SIZE`=8 here,
it is because in google colab, it will run into a memory issue with bert model
at max_length = 512. Usually, the batch_size can be set as 32. USE_XLA and
USE_AMP are two methods to help the train speed, in this notebook, we will not
discuss that. Therefore, we set them to False.


{% highlight python %}
BATCH_SIZE = 8

EVAL_BATCH_SIZE = BATCH_SIZE * 2

# XLA is the optimizing compiler for machine learning
# It can potentially increase speed by 15% with no source code changes
USE_XLA = False

# mixed precision results on https://github.com/huggingface/transformers/tree/master/examples
# Mixed precision can help to speed up training time
USE_AMP = False
{% endhighlight %}


{% highlight python %}
tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})
{% endhighlight %}


{% highlight python %}
# Steps is determined by the number of examples
import json
with open('data/info.json') as json_file:
    data_info = json.load(json_file)
    
train_examples = data_info['train_length']
valid_examples = data_info['validation_length']
test_examples = data_info['test_length']

train_examples, valid_examples, test_examples
{% endhighlight %}




    (4000, 1000, 5000)



Now that we have a pipeline setup, we need to start the process of converting
words into numbers so that they can be processed by the BERT transfer learning
backbone. This process is commonly called Tokenization, and Huggingface includes
a tokenizer that helps with this process.

The tokenizers are based on the underlying research code. For example, the
following are different BERT models that can be utilized within the BERT
framework:

* ``bert-base-uncased``: 12-layer, 768-hidden, 12-heads, 110M parameters
* ``bert-large-uncased``: 24-layer, 1024-hidden, 16-heads, 340M parameters
* ``bert-base-cased``: 12-layer, 768-hidden, 12-heads , 110M parameters
* ``bert-large-cased``: 24-layer, 1024-hidden, 16-heads, 340M parameters

As seen above, the different numbers have different levels of complexity and are
associated either with uncapitalized text (uncased) or text that has
capitalization. I selected the bert-base-cased underlying model because the
movie text have capitalization. I also selected it because in general models
that are less complex tend to run faster than models which are more complex.

The Transformers framework can use a configuration dictionary in order to set up
the hyperparameters for the model. In this case, I explictly use the config to
make sure that the model is looking at num_labels=2. If we had been going
through an example with three categories ('Positive', 'Negative', and 'Neutral')
as opposed to just two cases ('Positive' and 'Negative') then we would have
wanted to use num_labels=3 instead. [Ref](https://github.com/huggingface/transfo
rmers/blob/master/examples/run_tf_glue.py)


{% highlight python %}
# Load tokenizer and model from pretrained model/vocabulary. Specify the number of labels to classify (2+: classification, 1: regression)
num_labels = 2 
config = BertConfig.from_pretrained("bert-base-cased", num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', config=config)
{% endhighlight %}


{% highlight python %}
# Make use of the following config parameters

# {
#   "architectures": [
#     "BertForMaskedLM"
#   ],
#   "attention_probs_dropout_prob": 0.1,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "max_position_embeddings": 512,
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,
#   "type_vocab_size": 2,
#   "vocab_size": 28996
# }
{% endhighlight %}

Now that we have a tokenizer and the model with the right configurations, we
need to take our parsed tensors (the tr_parse_ds or train parsed dataset) and
feed them into the Huggingface framework. To do this, we are going to make a
slight modification to the glue_convert_examples_to_features code found in the
HuggingFace transformers repo. Here we are going to use the sst-2 task (the
Stanford Sentiment Treebank binary classification task) because this task also
works with binary classification.

Notes:

Huggingface uses the similar strategy of taking the TFExamples and using a
dataset in order to convert the "sentence" and "labels" into inputs that are
needed by BERT (inputs such as 'input_ids', 'attention_mask', and
'token_type_ids'). As disscussed earlier in the workbook, this transformation
process makes it quick to test out the conversion on a couple of data points and
to move onto the next step without waiting for the full conversion to complete.


{% highlight python %}
import time
start_time = time.time()
train_dataset = glue_convert_examples_to_features(examples=tr_clean_ds, tokenizer=tokenizer
                                                  , max_length=512, task='sst-2',
                                                  label_list=['0','1']
                                                  )
print(f"---{time.time()-start_time} seconds---")
{% endhighlight %}

    ---14.97620701789856 seconds---



{% highlight python %}
import time
start_time = time.time()
valid_dataset = glue_convert_examples_to_features(examples=val_clean_ds, tokenizer=tokenizer
                                                  , max_length=512, task='sst-2'
                                                  , label_list =['0', '1'])
print(f"---{time.time()-start_time} seconds---")
{% endhighlight %}

    ---3.7171196937561035 seconds---



{% highlight python %}
train_dataset = train_dataset.shuffle(train_examples).batch(BATCH_SIZE).repeat(-1)

valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
{% endhighlight %}

In this next section, we need to configure the loss function, the optimizer, and
any additional metrics that we want to capture.

Loss:

This is the objective that the model is trying to minimize. In our example, the
HuggingFace framework compares this against the distribution of predicted
classes. Said differently, we are trying to compare how similar the actual
distribution is to the predicted distribution, and this is captured in the loss
function called SparseCategoricalCrossentropy .

A good discussion on cross entropy can be found at The Gradient

Optimizer:

Deep learning is the process of minimizing a loss function. The process that
determines what steps to try out in each iteration is commonly referred to as
the optimizer. For this exercise, I used the Adam optimization algorithm which
tends to work well in a variety of situations.

Metric:

As a basic metric, we should look at the number of times that the actual class
is identical to what is predicted.

Unfortunately ,the model currently generates unscaled outputs for each example
(an unscaled output for the negative class and another unscaled output for the
positive class). Said differently, the model generates outputs before they are
converted into probabilities (the conversion happens with a softmax function).
Because of all of this, the appropriate metric to use would be
SparseCategoricalAccuracy.


{% highlight python %}
opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)

if USE_AMP:
    # loss scaling is currently required when using mixed precision
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=opt, loss=loss, metrics=[metric])
{% endhighlight %}

The TensorFlow documentation states the following:

**If x is a tf.data dataset, and 'steps_per_epoch' is None, the epoch will run
until the input dataset is exhausted.**

Because these datasets can be a precursor to training on TFRecords of
significant size, it is best practice to specificially state the number of steps
that will be processed per epoch in the train and validation stage.


{% highlight python %}
train_steps = train_examples//BATCH_SIZE
valid_steps = valid_examples//EVAL_BATCH_SIZE
{% endhighlight %}

GPUs run up to 27x faster that CPUs for model training. Because of this, it is
critical that the following preconditions are in place:

You are using the version of Tensflow for GPUs
The code has access to a GPU
To confirm the preconditions, I run the following code to detect GPUs and to see
the physical devices that are available.


{% highlight python %}
# GPU USAGE
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices()
{% endhighlight %}

    Num GPUs Available:  1





    [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
     PhysicalDevice(name='/physical_device:XLA_CPU:0', device_type='XLA_CPU'),
     PhysicalDevice(name='/physical_device:XLA_GPU:0', device_type='XLA_GPU'),
     PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]




{% highlight python %}
model.summary()
{% endhighlight %}

    Model: "tf_bert_for_sequence_classification_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    bert (TFBertMainLayer)       multiple                  108310272 
    _________________________________________________________________
    dropout_75 (Dropout)         multiple                  0         
    _________________________________________________________________
    classifier (Dense)           multiple                  1538      
    =================================================================
    Total params: 108,311,810
    Trainable params: 108,311,810
    Non-trainable params: 0
    _________________________________________________________________



{% highlight python %}
history = model.fit(train_dataset, epochs=3, steps_per_epoch=train_steps,
                    validation_data=valid_dataset, validation_steps=valid_steps)
{% endhighlight %}

    Train for 500 steps, validate for 62 steps
    Epoch 1/3
    500/500 [==============================] - 561s 1s/step - loss: 0.4208 - accuracy: 0.7883 - val_loss: 0.2677 - val_accuracy: 0.8942
    Epoch 2/3
    500/500 [==============================] - 543s 1s/step - loss: 0.2023 - accuracy: 0.9268 - val_loss: 0.2334 - val_accuracy: 0.9133
    Epoch 3/3
    500/500 [==============================] - 542s 1s/step - loss: 0.1015 - accuracy: 0.9672 - val_loss: 0.2833 - val_accuracy: 0.9103


Train for 500 steps, validate for 62 steps
Epoch 1/3
500/500 [==============================] - 561s 1s/step - loss: 0.4208 -
accuracy: 0.7883 - val_loss: 0.2677 - val_accuracy: 0.8942
Epoch 2/3
500/500 [==============================] - 543s 1s/step - loss: 0.2023 -
accuracy: 0.9268 - val_loss: 0.2334 - val_accuracy: 0.9133
Epoch 3/3
500/500 [==============================] - 542s 1s/step - loss: 0.1015 -
accuracy: 0.9672 - val_loss: 0.2833 - val_accuracy: 0.9103

It tooks about 30mins to train 3 epochs on this setting. And we can acctich 0.91
accuracy on mini dev set.

### Evaluate the results of the model


{% highlight python %}
import time
start_time = time.time()
test_dataset = glue_convert_examples_to_features(examples=test_clean_ds, tokenizer=tokenizer
                                                  , max_length=512, task='sst-2'
                                                  , label_list =['0', '1'])
print(f"---{time.time()-start_time} seconds---")
{% endhighlight %}

    ---19.080271005630493 seconds---



{% highlight python %}
test_dataset = test_dataset.batch(EVAL_BATCH_SIZE)

{% endhighlight %}


{% highlight python %}
model.evaluate(test_dataset)
{% endhighlight %}

    63/63 [==============================] - 42s 667ms/step - loss: 0.2860 - accuracy: 0.9090





    [0.2859640326868329, 0.909]



63/63 [==============================] - 42s 667ms/step - loss: 0.2860 -
accuracy: 0.9090
[0.2859640326868329, 0.909]

As you can see, with only using the small sample size, we can acheive 0.909
accurarcy on test set. It was reported 0.935 in orginal BERT paper in this base
model.

### Create a Confusion Matrix

We can also visualize the evaluation in terms of how many true positives, true
negatives, false positives, and false negatives occur. This visualization is
commonly called a Confusion Matrix.

Creation of the confusion matrix involves the following steps:

1) Take the unnormalized outputs from the model (the logits) and compress them
into probabilities (that by definition are between 0 and 1). The function that
converts logits into probabilities is the softmax function.

2) Once you have probabilities for each prediction, identify predicted emotion
('Negative' or 'Positive') by selecting the probability with the largest value.
This is accomplished with the argmax function.


{% highlight python %}
y_pred = tf.nn.softmax(model.predict(test_dataset))
{% endhighlight %}


{% highlight python %}
y_pred_argmax = tf.math.argmax(y_pred, axis=1)
{% endhighlight %}


{% highlight python %}
y_true = tf.Variable([], dtype=tf.int64)

for features, label in test_dataset.take(-1):
    y_true = tf.concat([y_true, label], 0)
{% endhighlight %}


{% highlight python %}
%matplotlib inline  
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

def visualize_confusion_matrix(y_pred_argmax, y_true):
    """

    :param y_pred_arg: This is an array with values that are 0 or 1
    :param y_true: This is an array with values that are 0 or 1
    :return:
    """
    
    cm = tf.math.confusion_matrix(y_true, y_pred_argmax).numpy()
    con_mat_df = pd.DataFrame(cm)
    
    print(classification_report(y_pred_argmax, y_true))
    
    sns.heatmap(con_mat_df, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print(classification_report(test_labels, baseline_predicted))
visualize_confusion_matrix(y_pred_argmax, y_true)
{% endhighlight %}


![png](data:image/png;base64,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
AWVmZqn0/wFS8bHOjn6W5AAAAABJRU5ErkJggg==
)


### Save model


{% highlight python %}
tf.saved_model.save(model, './202002')
{% endhighlight %}

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1781: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    INFO:tensorflow:Assets written to: ./202002/assets


## How-to-use-Saved-Model

In the previous tutorial, we looked at building a sentiment classifier for movie
reviews. At the end of that exercise we saved our model so that we can reuse it.

As seen in the command below, the SavedModel requires attention_mask, input_ids,
and token_type_ids as inputs. These are the inputs that are required by the
Google BERT model that we are using. Lucky for us, we can use the HuggingFace
Transformers class to convert a sentence into the required inputs.


{% highlight python %}
!saved_model_cli show --dir ./202002 --tag_set serve --signature_def serving_default
{% endhighlight %}

    The given SavedModel SignatureDef contains the following input(s):
      inputs['attention_mask'] tensor_info:
          dtype: DT_INT32
          shape: (-1, 128)
          name: serving_default_attention_mask:0
      inputs['input_ids'] tensor_info:
          dtype: DT_INT32
          shape: (-1, 128)
          name: serving_default_input_ids:0
      inputs['token_type_ids'] tensor_info:
          dtype: DT_INT32
          shape: (-1, 128)
          name: serving_default_token_type_ids:0
    The given SavedModel SignatureDef contains the following output(s):
      outputs['output_1'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 2)
          name: StatefulPartitionedCall:0
    Method name is: tensorflow/serving/predict


The following commands are going to load our model and the tokenizer which
converts words into numbers.


{% highlight python %}
savedmodel = tf.saved_model.load('./202002')

{% endhighlight %}


{% highlight python %}
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

{% endhighlight %}

The next step in this process is to create something that the Transformers
library can process. For our example, we are going to create a dictionary with
the required tensors, feed that dictionary into a data pipeline, and have the
Transformers library generate input based on the pipeline.


{% highlight python %}
example = {'idx': tf.constant(1, dtype=tf.int64), 'label': tf.constant(0, dtype=tf.int64) ,
           'sentence': tf.constant('This is the best store that I have ever visited', dtype=tf.string)}
{% endhighlight %}


{% highlight python %}
ds = tf.data.Dataset.from_tensors(example)
feature_ds = glue_convert_examples_to_features(ds, tokenizer, max_length=128, task='sst-2')
feature_dataset = feature_ds.batch(1)
{% endhighlight %}

Great! Now we have features in the format required by the Google BERT model. The
following function is going to convert these features into an actual prediction.


{% highlight python %}
def predict_dataset(feature_dataset, savedmodel):
    """
    :param feature_dataset: Contains information needed for BERT
    :param savedmodel: This is the model that has been pretrained in a sep process.
    :return: JSON output with the predicted classification. 
    """
    
    json_examples = []
    for feature_batch in feature_dataset.take(-1):
        feature_example = feature_batch[0]
    
        # The SavedModel is going to generate log probabilities (logits) as to whether the sentence
        # is negative (0) or positive (1).
        logits = savedmodel.signatures["serving_default"](attention_mask=feature_example['attention_mask'],
                            input_ids=feature_example['input_ids'],
                            token_type_ids=feature_example['token_type_ids'])['output_1']
        print(f"logits {logits}")
        
        # It is more helpful to have the actual probabilities of success. The TensorFlow softmax 
        # function will convert the logits into probabilities.
        probs = tf.nn.softmax(logits)
        
        # At this point we have probabilities (probs) of whether the sentence is negative or positive. 
        # These probabilites (by definition) will always sum to 100%.
        
        # It would be better though if we could just report out which probability is higher. 
        # This is done with the argmax function.
        
        prediction = tf.math.argmax(probs, axis=1)
    
        print(f"probs {probs}")
        print(f"prediction {prediction}")
    
        json_example = {"SENTIMENT_PREDICTION": str(prediction.numpy()[0])}
        json_examples.append(json_example)
    
    return json_examples
{% endhighlight %}


{% highlight python %}
negative_example = {'idx': tf.constant(1, dtype=tf.int64), 'label': tf.constant(0, dtype=tf.int64) ,
                    'sentence': tf.constant('This store is absolutely horrible and I hate it!!',
                                            dtype=tf.string)}
{% endhighlight %}


{% highlight python %}
negative_example

{% endhighlight %}




    {'idx': <tf.Tensor: id=313431, shape=(), dtype=int64, numpy=1>,
     'label': <tf.Tensor: id=313432, shape=(), dtype=int64, numpy=0>,
     'sentence': <tf.Tensor: id=313433, shape=(), dtype=string, numpy=b'This store is absolutely horrible and I hate it!!'>}




{% highlight python %}
def predict(example, tokenizer, savedmodel):
    """

    :param example: This is a single dictionary of tensors which contains a idx, a label, and a sentence
    :return: The prediction in JSON format. 1 is positive, and 0 is negative.
    """
    # The Transformers glue_convert_examples_to_features works well with datasets. 
    # It does not work well with a dictionary of examples. 
    ds = tf.data.Dataset.from_tensors(example)
    
    # Use the transformers library in order to convert an English sentence into something that 
    # BERT recognizes.
    
    # The conversion requires giving a label (even if we don't have one). The e-asiest way to get around this is to get around
    # this is to assign a default label of zero when you don't have a label. 
    
    feature_ds = glue_convert_examples_to_features(ds, tokenizer, max_length=512, task='sst-2')
    
    feature_dataset = feature_ds.batch(64)
    json_examples = predict_dataset(feature_dataset, savedmodel)
    
    return json_examples
{% endhighlight %}


{% highlight python %}
json_result = predict(negative_example, tokenizer, savedmodel)
{% endhighlight %}


{% highlight python %}
predict(example, tokenizer, savedmodel)
{% endhighlight %}

# Base Line

You may wonder how does a baseline model perform on this dataset. Below are only
couple lines code using BOW(bag of words) and Logistic Regression.


{% highlight python %}
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
{% endhighlight %}


{% highlight python %}
train_texts, train_labels = [row[1] for row in train_csv], [row[2] for row in train_csv]
test_texts, test_labels =  [row[1] for row in test_csv], [row[2] for row in test_csv]
{% endhighlight %}


{% highlight python %}
len(train_texts) ,  len(train_labels)
{% endhighlight %}




    (4000, 4000)




{% highlight python %}
baseline_model = make_pipeline(CountVectorizer(ngram_range=(1,3)), LogisticRegression()).fit(train_texts, train_labels)
{% endhighlight %}

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)



{% highlight python %}
baseline_predicted = baseline_model.predict(test_texts)
{% endhighlight %}


{% highlight python %}
print(classification_report(test_labels, baseline_predicted))
{% endhighlight %}

                  precision    recall  f1-score   support
    
               0       0.85      0.86      0.86      2475
               1       0.86      0.86      0.86      2525
    
        accuracy                           0.86      5000
       macro avg       0.86      0.86      0.86      5000
    weighted avg       0.86      0.86      0.86      5000


​    


{% highlight python %}
visualize_confusion_matrix(baseline_predicted,test_labels)
{% endhighlight %}


![png](data:image/png;base64,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
pf8PTWe4O0dC/DYAAAAASUVORK5CYII=
)

