
# coding: utf-8
# TODO: Document properly

# Requires tf 1.4+
# Cues from:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/predictor

# In[35]:


import tensorflow as tf
import tensorflow.contrib as tcb
from tensorflow.contrib.learn.python.learn.learn_io.numpy_io import numpy_input_fn
import time
import numpy as np


# In[36]:


X = np.linspace(-5,5)
Y = np.linspace(-5,5)
X, Y = np.meshgrid(X, Y)
Z = 100*np.exp(-1*(X**2 + Y**2)/5)
X1 = X.reshape(-1,1)
Y1 = Y.reshape(-1,1)
Z1 = Z.reshape(-1,1)


# In[56]:


def model_fn(features, labels, mode, params={}):
    X = features['X']
    Y = features['Y']
    
    inputs = tf.concat([X, Y], axis=1)
    hs = 10
    predictions = tcb.layers.fully_connected(inputs, num_outputs=hs, weights_initializer=None, activation_fn=tf.nn.relu)
    predictions = tcb.layers.fully_connected(predictions, num_outputs=1, weights_initializer=None, activation_fn=None)
    shift = tf.constant(params.get('shift'), dtype=tf.float64)

    if mode == tcb.learn.ModeKeys.INFER:
        return tcb.learn.ModelFnOps(
            mode=mode,
            predictions=predictions+shift,
            loss=None,
            train_op=None
        )
    
    loss = tf.losses.mean_squared_error(labels, predictions)
    optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)
    global_step = tf.contrib.framework.get_global_step()
    train_op = tf.group(optimizer, tf.assign_add(global_step, 1))
    
    return tcb.learn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


# In[57]:


input_fn = numpy_input_fn(x= {"X": X1, "Y": Y1}, y = Z1, batch_size=500, num_epochs=10, shuffle=False)


# In[59]:


estimator = tcb.learn.Estimator(model_fn=model_fn, params={'shift': 0}, model_dir="/tmp/v4")


# In[60]:


estimator.fit(input_fn=input_fn)


# In[61]:


# res = estimator.predict(input_fn=input_fn)
# for i in res:
#     print i


# In[62]:


from tensorflow.contrib import predictor


# In[63]:


tf.__version__


# In[64]:


def serving_input_fn():
    x = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='x')
    y = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='y')
    features = {'X':x, 'Y':y}
    return tcb.learn.utils.input_fn_utils.InputFnOps(features, None, default_inputs=features)


# In[69]:


estimator = tcb.learn.Estimator(model_fn=model_fn, params={'shift': 100}, model_dir="/tmp/v4")
estimator.export_savedmodel("export2", serving_input_fn=serving_input_fn, as_text=True)
estimator.export_savedmodel("export2txt", serving_input_fn=serving_input_fn, as_text=True)


# In[70]:


# Shift is a parameter to play with
Xq = np.array([1.0, 2.0], dtype=np.float64).reshape(-1,1)
Yq = np.array([1.0, 2.0], dtype=np.float64).reshape(-1,1)
input_fn = numpy_input_fn(x= {"X": Xq, "Y": Yq}, batch_size=500, num_epochs=1, shuffle=False)
res = estimator.predict(input_fn=input_fn)
for i in res:
    print i

