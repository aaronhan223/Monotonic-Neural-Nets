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


for fn in range(4):
    print('\n')
    print(now(), fn)
    ops.reset_default_graph()

    data, n_sample, lower, upper = load_data(n=100, lower=-2, upper=2, fn=fn)

    X_train = data['data_train'][:, data['X_cols']].astype(np.number)
    X_test  = data['data_test' ][:, data['X_cols']].astype(np.number)
    Y_train = data['data_train'][:, data['Y_col']:data['Y_col'] + 1].astype(np.number)
    Y_test  = data['data_test' ][:, data['Y_col']:data['Y_col'] + 1].astype(np.number)

    Y_train_orig = Y_train.copy()
    Y_test_orig  = Y_test.copy()
    beta = None
    # i.e. train on the residuals
    if with_linear:
        beta = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ Y_train)
        Y_train = Y_train - X_train @ beta
        Y_test = Y_test - X_test @ beta
    n, x_dim = X_train.shape

    # setup iterator
    full_batch = n_sample
    X_tf = tf.placeholder(tf.float32, [None, x_dim])
    Y_tf = tf.placeholder(tf.float32, [None, 1])
    ds = tf.data.Dataset.from_tensor_slices({'X': X_tf, 'Y': Y_tf})
    ds = ds.repeat(1)
    # ds = ds.shuffle(buffer_size=batch_size * 2)
    # ds = ds.padded_batch(batch_size=batch_size, padded_shapes={'X': [None], 'Y': [None]}, drop_remainder=True)
    ds = ds.batch(batch_size=full_batch)
    ds = ds.prefetch(full_batch)
    iterator = ds.make_initializable_iterator()
    next_element = iterator.get_next()
    X_trn, Y_trn = [next_element[key] for key in ['X', 'Y']]

    # full_ds = tf.data.Dataset.from_tensor_slices({'X': X_tf, 'Y': Y_tf})
    # full_ds = full_ds.repeat(1)
    # full_ds = full_ds.batch(batch_size=full_batch)
    # full_ds = full_ds.prefetch(full_batch)
    # full_iterator = full_ds.make_initializable_iterator()

    batch_factor = int(n / full_batch)
    n, x_dim = X_train.shape

    # not monotonic
    fnn = Monotone_Feedforward_Neural_Network(
        x_dim=x_dim,
        activate_last_layer=False,
        enforce_monotonicity=False,  # what matters
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
        'Mono. NN' + lin: {'model': mnn, 'X': [], 'Y': [], 'Y_pred': [], 'marker': 'g--'},
        'Random smoothing NN' + lin: {'model': fnn, 'X': [], 'Y': [], 'Y_pred': [], 'marker': 'y'}
    }

    with tf.Session() as sess:
        # train each model
        for model in results:
            curr = results[model]['model']
            for epoch in range(n_epochs):
                # Step 1, initialize dataset
                sess.run(iterator.initializer, feed_dict={X_tf: X_train, Y_tf: Y_train})
                # Step 2, train models
                MSE_loss = curr.fit(sess=sess, learning_rate=1e-2)
        # TODO: sigma should not be 0 when no appropriate, should make some judgement to choose the most appropriate sigma!
        sigma_space, sigma, bounded = np.logspace(-2, 2), 1 / abs(beta[0][0]), True
        if fn == 1:
            sigma = 5
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

            if 'smoothing' in model:
                cumsum, n_repeat = np.empty((n_sample, 1)), 500
                for i in range(n_repeat):
                    rand_noise = np.zeros_like(X_test) if i == 0 else np.random.normal(0, sigma, X_test.shape)
                    # TODO: modify this, if run on iterator.initializer, then feed for X_tf and Y_tf are mini-batches
                    sess.run(iterator.initializer, feed_dict={X_tf: X_test + rand_noise, Y_tf: Y_test_orig})
                    X_test_batch, Y_test_batch, y_pred_batch = sess.run([m.X_tf, m.Y_tf, m.Yhat_tf])
                    # the true prediction includes linear
                    if with_linear:
                        y_pred_batch = y_pred_batch + X_test_batch @ beta
                    if i == 0:
                        res['X'] = X_test_batch.reshape(-1, 1)
                        res['Y'] = Y_test_batch.reshape(-1, 1)
                        # res['Y_pred'].append(y_pred_batch.reshape(-1, 1))
                        cumsum = np.zeros_like(y_pred_batch.reshape(-1, 1))
                    else:
                        # print(np.average(np.squeeze(y_pred_batch.reshape(-1, 1) - res['Y'])))
                        cumsum += y_pred_batch.reshape(-1, 1)
                res['Y_pred'] = cumsum / (n_repeat - 1)

            else:
                sess.run(iterator.initializer, feed_dict={X_tf: X_test, Y_tf: Y_test_orig})
                X_test_batch, Y_test_batch, y_pred_batch = sess.run([m.X_tf, m.Y_tf, m.Yhat_tf])
                if with_linear:
                    y_pred_batch = y_pred_batch + X_test_batch @ beta
                res['X'] = X_test_batch.reshape(-1, 1)
                res['Y'] = Y_test_batch.reshape(-1, 1)
                res['Y_pred'] = y_pred_batch.reshape(-1, 1)

    coor_i, coor_j = fn // 2, fn % 2
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

plt.savefig('./smooth_plot.pdf')
