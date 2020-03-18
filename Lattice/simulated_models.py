import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python.framework import ops
from simulated_data import load_data
from matplotlib import pyplot as plt
import pickle as pk
import tensorflow
import tensorflow_lattice as tfl
from tensorflow import feature_column as fc
from utils import compute_quantiles
with_linear = False

fig, ax = plt.subplots(2,2,figsize=(10,10))
markers = ['r', 'b']
for fn in range(4):
    ops.reset_default_graph()
    results = {}
    for i,with_linear in enumerate([False, True]):
        if with_linear:
            lin = " + Linear Function"
        else:
            lin = ""

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
      
        min_label, max_label = float(np.min(Y_train)), float(np.max(Y_train))
        print(with_linear, min_label,max_label)
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

        feature_configs = [
            tfl.configs.FeatureConfig(
                name='x',
                lattice_size=1,# pwlcalibrator?
                monotonicity='increasing',
                # We must set the keypoints manually.
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    X_train[:,0],
                    num_keypoints=5)
            ),
            tfl.configs.FeatureConfig(
                name='z',
                lattice_size=1,# pwlcalibrator?
                monotonicity='increasing',
                # We must set the keypoints manually.
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    np.zeros(X_train[:,0].shape),
                    num_keypoints=5)
            )

        ]
        model = tf.keras.models.Sequential()
        model.add(
            tfl.layers.PWLCalibration(
               # Input keypoints for the piecewise linear function
               input_keypoints=compute_quantiles(
                    X_train[:,0],
                    num_keypoints=5),
               # Output is monotonically increasing in this feature
               monotonicity='increasing',
               # This layer is feeding into a lattice with 2 vertices
               output_min=min_label,
               output_max=max_label)
        )
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()],
            optimizer=tf.keras.optimizers.Adam(1e-2)
        )
        model.fit(
            X_train, Y_train, epochs=25, batch_size=batch_size, verbose=False
        )
        pred = model.predict(X_test) 
        if with_linear:
            pred += X_test @ beta 
        results['1D Lattice' + lin] = {'model': model, 
                                       'X': X_test, 
                                       'Y': Y_test, 
                                       'Y_pred': pred, 
                                       'marker': markers[i]}
    X_test  = data['data_test' ][:, data['X_cols']].astype(np.number)
    Y_test  = data['data_test' ][:, data['Y_col']:data['Y_col']+1].astype(np.number)

    coor_i, coor_j = fn//2, fn%2
    curr_ax = ax[coor_i, coor_j]
    curr_ax.scatter(X_test,Y_test)
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
