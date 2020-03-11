import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python.framework import ops
from lib import *
from lib.neuralnetwork import Monotone_Feedforward_Neural_Network
from lib.general_functions import now
from lib.metrics import accuracy_np, discrimination_np, consistency_np
from lib.metrics import k_nearest_neighbors_sp, identify_monotonic_pairs
from lib.metrics import resentment_individual, resentment_pairwise
from lib.metrics import lipschitz_sample_estimate
from lib.simulated_data import load_data
from matplotlib import pyplot as plt
import pickle as pk


with_linear = True

fig, ax = plt.subplots(2,2,figsize=(10,10))

for fn in range(4):
    print(now(), fn)
    ops.reset_default_graph()

    data = load_data(fn)
    
    X_train = data['data_train'][:, data['X_cols']].astype(np.number)
    X_test  = data['data_test' ][:, data['X_cols']].astype(np.number)
    Y_train = data['data_train'][:, data['Y_col']:data['Y_col']+1].astype(np.number)
    Y_test  = data['data_test' ][:, data['Y_col']:data['Y_col']+1].astype(np.number)
    
    Y_train_orig = Y_train.copy()
    Y_test_orig  = Y_test.copy()
    beta = None
    # i.e. train on the residuals
    if with_linear:
        beta = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ Y_train)
        Y_train = Y_train - X_train @ beta
        Y_test  = Y_test  - X_test @ beta
  
    n, x_dim = X_train.shape

    # setup iterator
    batch_size = 64
    X_tf = tf.placeholder(tf.float32, [None, x_dim])
    Y_tf = tf.placeholder(tf.float32, [None, 1])
    ds = tf.data.Dataset.from_tensor_slices({'X': X_tf, 'Y': Y_tf})
    ds = ds.repeat(1)
    ds = ds.shuffle(buffer_size=batch_size * 2)
    #ds = ds.padded_batch(batch_size=batch_size, padded_shapes={'X': [None], 'Y': [None]}, drop_remainder=True)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(batch_size)
    iterator = ds.make_initializable_iterator()
    next_element = iterator.get_next()
    X_trn, Y_trn = [next_element[key] for key in ['X', 'Y']]

    batch_factor = int(n / batch_size)
    n, x_dim = X_train.shape

    # not monotonic
    fnn = Monotone_Feedforward_Neural_Network(
        x_dim = x_dim,
        activate_last_layer=False,
        enforce_monotonicity=False, # what matters
        X_tf=X_trn,
        Y_tf=Y_trn,
        var_scope='ffnn'
    )
    # monotonic
    mnn = Monotone_Feedforward_Neural_Network(
        x_dim = x_dim,
        monotonicity = data['monotonicity'],
        activate_last_layer=False,
        enforce_monotonicity=True,
        X_tf=X_trn,
        Y_tf=Y_trn,
        var_scope='mnn'
    )
    
    n_epochs = 50
    if with_linear:
        lin = " + Linear Function"
    else:
        lin = ""
    results = {
        'Non-mono. NN' + lin: {'model': fnn, 'X': [], 'Y': [], 'Y_pred': [], 'marker': 'r'},
        'Mono. NN' + lin: {'model': mnn, 'X': [], 'Y': [], 'Y_pred': [], 'marker': 'g--'}
    }
    with tf.Session() as sess:
        # train each model
        for model in results:
            curr = results[model]['model']
            for epoch in range(n_epochs):
                # Step 1, initialize dataset
                sess.run(iterator.initializer, feed_dict={X_tf: X_train, Y_tf:Y_train})
                # Step 2, train models
                MSE_loss = curr.fit(
                    sess = sess,
                    learning_rate = 1e-2
                )
        # Step 3, After training, make predictions
        for model in results:
            print(model)
            res = results[model]
            m = res['model']
            sess.run(iterator.initializer, feed_dict={X_tf:X_test, Y_tf:Y_test_orig})
            for j in range(int(X_test.shape[0]/batch_size)+1):
                X_test_batch, Y_test_batch, y_pred_batch = sess.run([m.X_tf,m.Y_tf,m.Yhat_tf])
                # the true prediction includes linear
                if with_linear:
                    y_pred_batch = y_pred_batch + X_test_batch @ beta
                res['X'].append(X_test_batch.reshape(-1,1))
                res['Y'].append(Y_test_batch.reshape(-1,1))
                res['Y_pred'].append(y_pred_batch.reshape(-1,1))
    
    for model in results:
        res = results[model]
        res['X'] = np.vstack(res['X'])
        res['Y'] = np.vstack(res['Y'])
        res['Y_pred'] = np.vstack(res['Y_pred'])

    coor_i, coor_j = fn//2, fn%2
    curr_ax = ax[coor_i, coor_j]
    curr_ax.scatter(results[model]['X'], results[model]['Y'])
    for model in results:
        res = results[model]
        X_final = res['X']
        Y_pred = res['Y_pred']
        idx = np.argsort(X_final.squeeze())
        X_final = X_final[idx]
        Y_pred = Y_pred[idx]
        curr_ax.plot(X_final, Y_pred, res['marker'], label=model, linewidth=4)
    curr_ax.legend()

plt.show()
