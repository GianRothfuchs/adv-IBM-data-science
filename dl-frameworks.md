# Deep Learning Frameworks

## TensorFlow

TensorFlow crates a static computation graph. That can be executed on a large amount of data. A computation graph consists of computation nodes that are connected with data tensors, flowing from one node to the other. Hence, Tensor Flow. The computation graph is created by adding placeholders. A placeholder is a tensor where data is fed in during execution time. 
While the placeholder is meant to hold data a TensorFlow variable is tweaked during execution. A variable can be saved to disk during and after training. The following code creates a graph running 10 logistic regression models in parallel on an input that is a 1x784 vector. The following syntax refers to TensorFlow 1.X version and is depreciated:

```python

	x = tf.placeholder(tf.float32, [None,784]) # a placeholder
	W = tf.Variable(tf.zeros([784,10]))		# a variable		
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x,W) + b)   # prediction
	
	y_ = ft.placeholder(tf.float32,[None,10])
	
	cross_entropy = tf.reduce_mean(-tf.reduce_sum( y_ * tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	
	# upuntil here the computation graph has been defined, not much computation happened so far.
	# here's where execution starts:
	sess = tf.InteractiveSession()
	
	tf.global_variable_initializer().run()
	
	for _ in range(1000):
		batch_xs, batch_ys = mnist_next_batch(100)
		sess.run(train_steps, feed_dict={x: batch_xs, y_:batch_ys})
	
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print(sess.run(accuracy, feed_dict={x : mnist.test.images, y_ : mnist.test.labels}))
	
	
```

## TensorBoard

### Model Graph

The code generated model graph can be displayed in the model graph section of the TensorBoard. Where again the nodes depict computation nodes and the edges are tensors.


### Scalar Metrics: Loss & Accuaracy

TensorBoard provides a visual way to debug/inspect the execution of a TensorFlow implementation of a neural network. the behaviour of the loss function can be inspected to get an indication of the convergence behavior and how well the learning rate is set. 

[TensorBoard: loss function](https://raw.githubusercontent.com/tensorflow/tensorboard/master/docs/images/scalars_loss.png)

In TensorBoard it is possible to compare different runs in many dimensions for example in terms of of accuaracy and loss function behavior. It is considered a good sign when loss and accuaracy have inverse shape.

### Training Weights & Activations

The weight matrices can also be inspected in a graphical maner. Model degeneration where coefficients are all close to zero are considered problematic. Uniform distributions resmbling the random intialization indicates a layer did not learn anything. Both are considered as red flags. It may also proof useful to plot activations of the output layer nodes. 
The closer the more recent the run is. The x axis shos the distribution of values and the y axis the frequency.

[TensorBoard: weights](https://camo.githubusercontent.com/10514955347a3c38d186cc94ccd81bf8c6616d25d752c6cc281ed9842e2a5261/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74656e736f72626f6172642f686973746f6772616d5f64617368626f6172642f365f74776f5f646973747269627574696f6e732e706e67)

Further information on histograms in [TensorBoard](https://github.com/tensorflow/tensorboard/blob/master/docs/r1/histograms.md).


## Tensorflow 2.X

### Eager execution

Up until TensorFlow version 2.0 eager execution was switched of by default. The difference can be best shown by means of an example in the old world (prior to version 2.0) and a new world version 2.X. In the old world a session had to be created to execute (and find mistakes).

```python
import numpy as np

a = tf.constant(np.array([1.,2.,3.]))
type(a) # = tensorflow.python.frameworks.ops.Tensor

b = tf.constant(np.array([4.,5.,6.]))
type(b) # = tensorflow.python.frameworks.ops.Tensor

c = tf.tensordot(a,b,1)

print(c.numpy()) # doesn't work

# to evaluate/debug a session has to be created first, before you se anything 
session = tf.Session()
output = session.run(c)
print(output)

```

In the new world tahnks to eager execution intermediary results can be generated and inspected:

```python
import numpy as np

tf.__version__ # > 2.0.0-alpha0 

a = tf.constant(np.array([1.,2.,3.]))
type(a) # = tensorflow.python.frameworks.ops.EagerTensor (NEW!)

b = tf.constant(np.array([4.,5.,6.]))
type(b) # = tensorflow.python.frameworks.ops.EagerTensor

c = tf.tensordot(a,b,1)

print(c.numpy()) # works! (NEW!)

#session call not required

```

### Keras as Fist Class Citizen ion TensorFlow

Instad of approaching TensorFlow via Keras it is now possible to call Keras objects from TensorFlow directly in the following way:

```python
# this is the cool part (see text below)
ps_strategy = tf.distribute.experimental.ParameterServerStrategy()
with ps_strategy.scope:
	#indent itmes below
	# like 
	# model = tf...
	# model.compile(...
	# model.fit(...


# this is keras syntax in tf, activation funciton are form tf now
model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(128, activation=tf.nn.reul),
	tf.keras.layers.Flatten(10, activation=tf.nn.softmax),
])
# this is keras syntax in tf

model.compile(	optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5)

test_loss, test_acc = model.evaluate(test_images,test_labels)

```

The part above where tf.disribute.[...] is called actually disributes the model evaluation accross multiple computing nodes. The same code running on 1 node can be distributed to 1000 nodes. This is where the strength of TF can be combined with the convenience of Keras to build a model. The tf.keras.layers.(...) part can be skipped when the corresponding parts are imported the following way: from tensorflow.keras.layers import Flatten, Dense.


## Keras

### Overview
Keras is a high-level deep learnig library in Python that has been built on top of low-level engines like TensorFlow, Theano, CNTK. It helps to prototype deep learning models very quickly.

There are essentially two groups of models that can be created. Sequential models and non-sequetial models. the basic buildig block of a model is the layer, that can come in a large variety of forms. A squential model have an input and an output as well as input shape and output shape. Weights can be retrieved by the function layer.get_weights() as np arrays. Each layer has its configuration that can be retrieved by calling layer.get_config().

A schematic overview of the basic steps necessary to run a model in Keras can be found in the following. [Notebook](/notebooks/keras_intro.ipynb)

The basic steps performed to create as sqeutial model are the following:

1. Instantiate the model
```python
model = Sequential()
```
2. Add layers one by one using the .add() function.
```python
model.add(Dense(256, activation='sigmoid',input_shape=(784,)))
model.add(Dense(10, activation='sigmoid')) 
```
3. Compile the model with a loss function, optimizer and evaluation metric.
```python
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
```
4. Fit the model 
```python
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data= (x_test,y_test))
```

In Step 3 above in the compile step the optimizer  has been defined as a string 'sgd'. However, this leaveas us with just using the default implementation of SGD (sequential gradient descent). In case the default is not desired SGD can be modified and passed into the complier by doing the following:

```python
	from keras.optimizers import SGD
	# specify params
	sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9)
	# pass to compiler
	model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
```

### Example: Feed-Forward Neural Network

[Example Feed-forward NN](/notebooks/keras_intro_working.ipynb)

### Example: Recurrent Neural Network

There are a few very popular RNNs currently in use:
1. Basic RNN
2. GRU (Gated Recursive Unit)
3. LSTM (Long-Short-Term Memory)

The follwing only considers the LSTM. To use Keras to implement an LSTM model the following building Block is required:

```python
	from keras.layers.recurrent import LSTM
	
	tf.keras.layers.LSTM(units,
    activation="tanh",
    recurrent_activation="hard_sigmoid",
    recurrent_initializer="orthogonal",
    recurrent_regularizer=None,
    bias_regularizer=None,
    dropout=0.0,
    recurrent_dropout=0.0, # droput is defined within the layer
    return_sequences=False # returning intermediate values or not
	)
```

A useful tool to is embedding. This helps for example to cast a vocabulary into a vector of fixed size representation. 

```python
	from keras.layers.embeddings import Embedding
	
    Embedding(
    input_dim, 				 	# Volcabulary Size
    output_dim,					# Ouptut vector length
    embeddings_initializer="uniform",
    mask_zero=False)			# sentences of varying lengt can be made same length this way

```

[Example LSTM ](/notebooks/keras_intro_lstm.ipynb)

### Example Non-Sequential Models

in contrast to sequential model this kind of model accepts inputs and transforms those to one or more outputs. In the follwoing example input is consumed and then x is trasnformed an arbitrary number of times until it is outputed into predcitions:

```python
	from keras.layers import Input, Dense
	from keras.layers import Model
	
	num_classes = 10
	inputs = Input(shape=(784,))
	
	x = Dense(512, activation='relu')(inputs) 	#  : input -> x
	x = Dense(512, activation='relu')(x)	 	# x -> x
	
	predictions = Dense(num_classes, activation='softmax')(x) # x -> pred

	model = Model(inputs = inputs, outputs = predictions)
	model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
	model.fit(...)

```

To add a layer to the model a Dense(...) is called on inputs. The layer Dense() is called on a tensor, here inputs. 

### Saving and Loading Models

In Keras every model can be stored to disk. There are a couple of ways to achieve this goal. One way is the HDF5. HDF5 allows to save the model architecture, the weights and also the state of the model after training. This allows to pickup the model where it has been left off. It is also possible to just save the architecture (JSON/YAML) or the weights only (HDF5).

```python

from keras.models import model_from_json

# Saving model as JSON and weights as HDF5
json_string = model.to_json() # alternative: to_yaml()
model.save_weights('weights.h5')

# Load from JSON and set weights
model = model_from_json(json_string)
model.load_weights('weights.h5')

```

To save the entire model the following can be done:

```python
from keras.models import load_model

# Saving model 
model.save('full_model.h5')
# Loading model 
model = load_model('full_model.h5')
```






