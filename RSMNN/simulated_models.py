import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from lib.neuralnetwork import Monotone_Feedforward_Neural_Network
from lib.general_functions import now
from lib.simulated_data import load_data
from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb

tf.disable_v2_behavior()
with_linear = True

fig, ax = plt.subplots(2, 2, figsize=(10, 10))


def get_monotonicity_bounded(W, sigma):
    if W >= 1 / sigma:
        return True
    return False

# only for sigma progression
#fn = 1
#data, n_sample, lower, upper = load_data(n=100, lower=-2, upper=2, fn=fn)
#for k,sigma in enumerate([0.1,1,2.5,5]):
sigmas = [0.5, 0.5, 0.5, 0.5]
for k,fn in enumerate(range(4)):    
    #if fn != 1:
    #    continue
    print('\n')
    print(now(), fn)
    ops.reset_default_graph()

    data, n_sample, lower, upper = load_data(n=250, lower=-2, upper=2, fn=fn)

    X_train = data['data_train'][:, data['X_cols']].astype(np.number)
    X_test  = data['data_test' ][:, data['X_cols']].astype(np.number)
    Y_train = data['data_train'][:, data['Y_col']:data['Y_col'] + 1].astype(np.number)
    Y_test  = data['data_test' ][:, data['Y_col']:data['Y_col'] + 1].astype(np.number)

    Y_train_orig = Y_train.copy()
    Y_test_orig  = Y_test.copy()
    beta = None
    # i.e. train on the residuals
    if with_linear:
        X_train_bias = np.tile(X_train, (1,2))
        X_train_bias[:,0] **= 0 # make column of ones for bias
        beta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ (X_train_bias.T @ Y_train)
        beta_0 = beta[0]
        beta = beta[1:2].reshape(1,1)
        Y_train = Y_train - X_train @ beta - beta_0
        Y_test = Y_test - X_test @ beta - beta_0
    n, x_dim = X_train.shape

    # setup iterator
    batch_size = 64
    X_tf = tf.placeholder(tf.float32, [None, x_dim])
    Y_tf = tf.placeholder(tf.float32, [None, 1])
    ds = tf.data.Dataset.from_tensor_slices({'X': X_tf, 'Y': Y_tf})
    ds = ds.repeat(1)
    # ds = ds.shuffle(buffer_size=batch_size * 2)
    # ds = ds.padded_batch(batch_size=batch_size, padded_shapes={'X': [None], 'Y': [None]}, drop_remainder=True)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(batch_size)
    iterator = ds.make_initializable_iterator()
    next_element = iterator.get_next()
    X_trn, Y_trn = [next_element[key] for key in ['X', 'Y']]

    # full_ds = tf.data.Dataset.from_tensor_slices({'X': X_tf, 'Y': Y_tf})
    # full_ds = full_ds.repeat(1)
    # full_ds = full_ds.batch(batch_size=batch_size)
    # full_ds = full_ds.prefetch(batch_size)
    # full_iterator = full_ds.make_initializable_iterator()

    batch_factor = int(n / batch_size)
    n, x_dim = X_train.shape

    # not monotonic
    fnn = Monotone_Feedforward_Neural_Network(
        x_dim=x_dim,
        activate_last_layer=False,
        enforce_monotonicity=False,  # what matters
        X_tf=X_trn,
        Y_tf=Y_trn,
        var_scope='ffnn',
        smoothing=False
    )
    # monotonic
    mnn = Monotone_Feedforward_Neural_Network(
        x_dim = x_dim,
        monotonicity = data['monotonicity'],
        activate_last_layer=False,
        enforce_monotonicity=True,
        X_tf=X_trn,
        Y_tf=Y_trn,
        var_scope='mnn',
        smoothing=False
    )
    smooth_nn = Monotone_Feedforward_Neural_Network(
        x_dim=x_dim,
        activate_last_layer=False,
        enforce_monotonicity=False,  # what matters
        X_tf=X_trn,
        Y_tf=Y_trn,
        var_scope='smooth_nn',
        smoothing=True,
        n_steps=100
    )

    
    n_epochs = 100
    if with_linear:
        lin = " + Linear Function"
    else:
        lin = ""
    results = {
 #       'Linear Function': {'model': lambda x: x @ beta + beta_0, 'X': [], 'Y': [], 'Y_pred': [], 'marker': 'purple'},
#        'Non-mono. NN' + lin: {'model': fnn, 'X': [], 'Y': [], 'Y_pred': [], 'marker': 'r'},
#        'Mono. NN' + lin: {'model': mnn, 'X': [], 'Y': [], 'Y_pred': [], 'marker': 'g--'},
        'Random smoothing NN' + lin: {'model': smooth_nn, 'X': [], 'Y': [], 'Y_pred': [], 'marker': 'y'}
    }

    with tf.Session() as sess:
        # train each model
       # sigma_space, sigma, bounded = np.logspace(-2, 2), 1 / abs(beta[0][0]), True
        sigma = sigmas[k]
        print("sigma:", sigma)
        for model in results:
            if model == 'Linear Function':
                continue
            curr = results[model]['model']
            for epoch in range(n_epochs):
                # Step 1, initialize dataset
                sess.run(iterator.initializer, feed_dict={X_tf: X_train, Y_tf: Y_train})
                # Step 2, train models
                MSE_loss = curr.fit(sess=sess, learning_rate=1e-3, sigma=sigma)
            print(model, MSE_loss)
        # TODO: sigma should not be 0 when no appropriate, should make some judgement to choose the most appropriate sigma!
        # print('Finding appropriate sigma...\n')
        # for s in tqdm(sigma_space):
        #     if bounded:
        #         monotonicity = get_monotonicity_bounded(beta[0][0], s)
        #         sigma = s
        #         if monotonicity:
        #             break
        # print('Current slope: {}, required largest sigma: {}, current sigma: {}'.format(beta, 1 / beta, sigma))
        # pdb.set_trace()
        # Step 3, After training, make predictions
        for model in results:
            print(model)
            res = results[model]
            m = res['model']
            sess.run(iterator.initializer, feed_dict={X_tf: X_test, Y_tf: Y_test_orig})
            for j in range(int(X_test.shape[0]/batch_size)+1):
                if 'smoothing' in model:
                    noise = np.random.normal(0,sigma,m.n_steps)
#                    noise = np.zeros((m.n_steps))
                    X_test_batch, Y_test_batch, Y_hats, y_pred_batch = sess.run([m.X_tf, m.Y_tf, m.Y_hats,m.Yhat_tf], feed_dict={m.noise: noise})
                    if with_linear:
                        y_pred_batch = y_pred_batch + X_test_batch @ beta + beta_0
                    res['X'].append(X_test_batch.reshape(-1, 1))
                    res['Y'].append(Y_test_batch.reshape(-1, 1))
                    res['Y_pred'].append(y_pred_batch.reshape(-1, 1))
#                    print(y_pred_batch)
                elif model=='Linear Function':
                    res['X'].append(X_test.reshape(-1,1))
                    res['Y'].append(Y_test_orig.reshape(-1,1))
                    res['Y_pred'].append(m(X_test).reshape(-1,1))
                else:
                    X_test_batch, Y_test_batch, y_pred_batch = sess.run([m.X_tf, m.Y_tf, m.Yhat_tf])
                    if with_linear:
                        y_pred_batch = y_pred_batch + X_test_batch @ beta + beta_0
                    res['X'].append(X_test_batch.reshape(-1, 1))
                    res['Y'].append(Y_test_batch.reshape(-1, 1))
                    res['Y_pred'].append(y_pred_batch.reshape(-1, 1))
        writer = tf.summary.FileWriter("output", sess.graph)
        writer.close()

    for model in results:
        res = results[model]
        res['X'] = np.vstack(res['X'])
        res['Y'] = np.vstack(res['Y'])
        res['Y_pred'] = np.vstack(res['Y_pred'])

    coor_i, coor_j = k // 2, k % 2
    curr_ax = ax[coor_i, coor_j]
    curr_ax.scatter(results[model]['X'], results[model]['Y'])
    og_x = results[model]['X']
    for model in results:
        res = results[model]
        X_final = res['X']
        Y_pred = res['Y_pred']
        idx = np.argsort(X_final.squeeze())
        X_final = X_final[idx]
        Y_pred = Y_pred[idx]
        curr_ax.plot(X_final, Y_pred, res['marker'], label=model, linewidth=4)
    curr_ax.legend()
    curr_ax.set_title("Sigma = " + str(sigma))
#    break
plt.show()
#plt.savefig('./smooth_plot.pdf')
