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









