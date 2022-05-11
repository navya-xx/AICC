import time
import json
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import Callback
import keras.backend as K
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh

def custom_loss_nmse(y_true, y_pred):
    # return tf.norm(y_true - y_pred) / tf.norm(y_true)
    return tf.norm(y_true - y_pred, axis=-1) / tf.norm(y_true, axis=-1)

def custom_loss_nmse_numpy(y_true, y_pred):
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    
    return np.mean(np.linalg.norm(y_true - y_pred, axis=-1) / np.linalg.norm(y_true, axis=-1))

def custom_loss_reg(y_true, y_pred):
    return tf.norm(y_true - y_pred, axis=-1) + 5 * (tf.norm(y_pred, axis=-1) - 1.0)**2

def custom_eigvec_norm(y_true, y_pred):
    return (tf.norm(y_pred, axis=-1) - 1.0)**2


# design NN for training
def create_model(matrix_dim, output_dim, num_inputs, n_nodes, degrees, num_layers=2, activation="relu"):
    """Create training model for DNN.
    Implements computation function with inputs only from the encoder.

    Parameters
    ----------
    input_shape : int
        Shape of input into to DNN.
        For flattened matrix input, shape is N^2
    num_inputs : int
        number of input matrices
    n_nodes : int
        Num of neurons in each layer of DNN
    n_layers : int
        Num of layers in each DNN
    activation : str
        Activation function to use in each layer
    degrees : list
        List of degrees for the encoding and computation function
    kernel_reg_factor : float
        Kernel regularization factor for both kernel and bias terms

    Returns
    -------
    keras.Sequential.model
        DNN model for training.
    """

    e_degree = degrees[0]
    h_degree = degrees[1]

    inputs = []
    for i in range(num_inputs):
        inputs.append(keras.Input(shape=(matrix_dim * matrix_dim,), name=f'X{i}'))

    b = [keras.Input(shape=(1,), name='Betas')]
    inputs.append(b[0])
    for j in range(1, e_degree):
        b.append(layers.Multiply()([b[j-1], b[0]]))

    def my_dense(input_data, output_nodes, n_layers, name="my_dense"):
        output = layers.Dense(n_nodes, activation=activation, name="%s_l0"%name)(input_data)
        for l in range(n_layers):
            output = layers.Dense(n_nodes, activation=activation, name="%s_l%d"%(name, l+1))(output)
        output = layers.Dense(output_nodes, name="%s_out"%name)(output)
        return output


    sums = []
    input_stacked = layers.Concatenate(axis=-1)(inputs[:-1])
    input_shape = input_stacked.shape[1:]
    for e in range(1, e_degree+1):
        enc_dense = my_dense(input_stacked, matrix_dim**2, num_layers, name="Enc_d%d"%e)
        sums.append(enc_dense)

    coeff_dense = my_dense(input_stacked, matrix_dim**2, num_layers, name="Enc_d0")

    coeff_dense_1 = my_dense(input_stacked, output_dim, num_layers, name="Dec_d0")
        
    # Multiply each coeeficient by beta, beta^2,...beta^(e_degree)
    mult = []
    for a in range(0, e_degree):
        mult.append(layers.Multiply()([sums[a], b[a]]))
    mult.append(coeff_dense)

    # output of e(beta)
    sums_final = layers.Add()(mult)
    outputs_matrix = layers.Reshape((matrix_dim, matrix_dim), input_shape=(matrix_dim**2,))(sums_final)

    x_poly = [outputs_matrix]
    for c in range(1, h_degree + 1):
        x_poly.append(layers.Dot(axes=(2, 1))([x_poly[c-1], outputs_matrix]))

    poly_reshape = []
    for l in range(h_degree + 1):
        poly_reshape.append(layers.Reshape((matrix_dim**2, ), input_shape=(matrix_dim, matrix_dim))(x_poly[l]))
        
    out_dense = []
    for s in range(h_degree):
        out_dense.append(layers.Dense(output_dim)(poly_reshape[s]))

    out_dense.append(coeff_dense_1)
    outputs = layers.Add()(out_dense)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



def random_bulk_data(rng, matrix_dim, num_inputs, batch_size, params):
    """Same as `generate_data`, but only generate bulk data without repetitions.

    Parameters
    ----------
    rng : Seed for generating random inputs given distribution
    matrix_dim: int
        Dimension of input matrix
    data_len: int
        number of input training pairs
    params: dict
        dictionary that stores the extra parameters for data generation

    """

    m, s, dist, problem = params["dist_param_1"], params["dist_param_2"], params["distribution"], params["problem"]

    data_len = batch_size * num_inputs

    if dist == "gaussian":
        gen_data_fn = lambda out_size: rng.normal(m, s, size=out_size)
    elif dist == "uniform":
        gen_data_fn = lambda out_size: rng.uniform(m, s, size=out_size)
    else:
        raise NotImplementedError(
            "Only `gaussian`  and `uniform` distributions are supported." "`%s` is not supported" % dist
        )

    X = gen_data_fn((data_len, matrix_dim, matrix_dim))

    if problem == "eigenvalue":
        X = 0.5 * (X + np.transpose(X, [0, 2, 1]))  # symmetric
        y = np.sort(np.linalg.eigvalsh(X), axis=-1)

    elif problem == "max_eigenvec":

        def max_eigvec(X):
            _, eigvec = eigsh(X, 1, which='LM')
            # maxcol = list(eigval).index(max(eigval))
            ev = np.squeeze(eigvec)
            # assert (ev > 0).all() or (ev < 0).all()
            if (ev < 0).all():
                ev = ev * (-1)
            return ev / np.linalg.norm(ev)
        X = 0.5 * (X + np.transpose(X, [0, 2, 1]))  # symmetric
        y = np.zeros((data_len, matrix_dim))
        for i in range(data_len):
            y[i]  = max_eigvec(X[i])
    
    elif problem == "matrix_exp":
        X_norm = np.linalg.norm(X, ord=2, axis=(-2, -1), keepdims=True)
        X = X / X_norm
        # X = 0.5 * (X + np.transpose(X, [0, 2, 1]))  # symmetric
        y = np.zeros((data_len, matrix_dim * matrix_dim))
        for i in range(data_len):
            y[i]  = expm(X[i]).flatten()

    elif problem == "matrix_det":
        for i in range(data_len):
            X[i] = np.eye(matrix_dim) - (X[i] - np.diag(np.diag(X[i])))

        y = np.linalg.det(X)
        y = np.expand_dims(y, axis=-1)

    else:
        raise NotImplementedError(
            "Supported `problem`  are `eigenvalue`, `max_eigenvec`, `matrix_exp` and `matrix_det`." " `%s` is not supported." % problem
        )
    
    X = np.reshape(X, (batch_size, num_inputs, -1))
    y = np.reshape(y, (batch_size, num_inputs, -1))

    return X, y


def convert_data_tf(X, y, num_input, recovery_threshold, for_training=True):

    training_len = X.shape[0]
    
    betas = np.arange(1, num_inputs + 1) / num_inputs
    alphas = np.arange(1, recovery_threshold + 1) / (recovery_threshold + 1)

    if for_training:
        input_matrices = np.zeros((num_input, num_input * training_len, X.shape[-1]))
        outputs = np.zeros((num_input * training_len, y.shape[-1]))

        for i in range(training_len):
            for m in range(num_input):
                outputs[i * num_input + m] = y[i, m, :]
                for n in range(num_input):
                    input_matrices[m][i * num_input + n] = X[i, m, :]

        betas_tiled = np.tile(betas.reshape(-1, 1), (training_len, 1))

    else:
        input_matrices = np.zeros((num_input, recovery_threshold * training_len, X.shape[-1]))
        outputs = np.zeros((num_input * training_len, y.shape[-1]))
        
        for i in range(training_len):
            for m in range(num_input):
                outputs[i * num_input + m] = y[i, m, :]
                for n in range(recovery_threshold):
                    input_matrices[m][i * recovery_threshold + n] = X[i, m, :]

        betas_tiled = np.tile(alphas.reshape(-1, 1), (training_len, 1))

    inputs = [x for x in input_matrices]
    inputs.append(betas_tiled)

    return inputs, outputs


class DataGenerator(keras.utils.Sequence):
    def __init__(self, gen_params, len, recovery_threshold):
        self.params = gen_params
        self.len = len
        self.num_inputs = gen_params["num_inputs"]
        self.recovery_threshold = recovery_threshold
        # self.old_data = None

    def __getitem__(self, index):
        # my_print(self.params)

        X_data, y_data = random_bulk_data(**self.params)
        data_X, data_Y = convert_data_tf(X_data, y_data, self.num_inputs, self.recovery_threshold)

        return data_X, data_Y

    def __len__(self):
        return self.len

def lagrange_interpolator(alphas, betas, outputs):
    """Generate polynomial using Lagrange interpolation

    Parameters
    ----------
    alphas : list of floats [N,]
        Scalars used to generate the outputs via encoding-computation process
    betas : list of floats [K,]
        Scalars corresponding to the trained dataset of size K
    outputs : list of Arrays [T, N, M, M]
        Arrays corresponding to the output of encoder-computation using alpha scalars

    Returns
    -------
    list of Arrays [K, ...]
        List of arrays corresponding to the true function output
    """
    N = len(alphas)

    K = len(betas)

    def interp_poly(z):
        prod_seq = np.zeros((N,), dtype=float)
        for i in range(N):
            prod = 1.0
            for j in range(N):
                if i == j:
                    pass
                else:
                    prod *= (z - alphas[j]) / (alphas[i] - alphas[j])
            prod_seq[i] = prod
        # prod_seq = tf.squeeze(tf.constant(prod_seq, dtype=tf.float32))

        # retVal = np.tensordot(prod_seq, outputs, axes=(-1, 1))
        retVal = np.average(outputs, axis=1, weights=prod_seq)

        return retVal

    results = []
    for i in range(K):
        results.append(interp_poly(betas[i]))

    return np.stack(results, axis=1)


if __name__ == "__main__":

    ##%%
    import argparse

    parser = argparse.ArgumentParser(
        description="Find the largest eigenvalue of a matrix of given dimensions.",
        epilog="Train and test using DNN with tensorflow. "
        "Output is logged with tensorboard in a log folder specified as input.",
    )
    parser.add_argument("--matrix-dim", "-a", type=int, default=10, help="Dimension of square matrix")

    parser.add_argument(
        "--num-nodes",
        "-c",
        type=int,
        default="16",
        help="Number of NN nodes in each layer, everywhere."
    )
    parser.add_argument(
        "--log-dir",
        "-f",
        type=str,
        default="/data/datapool/users/agrawal/TFlogDir/",
        help="Location of the log directory.",
    )
    parser.add_argument(
        "--save-dir",
        "-g",
        type=str,
        default="/data/datapool/users/agrawal/TFsavedModel/",
        help="Location for saving the trained NN models.",
    )
    parser.add_argument("--train", action="store_true", help="Train the network if this flag is set.")
    parser.add_argument("--test", action="store_true", help="Test the network if this flag is set.")
    parser.add_argument(
        "--model-id", "-j", type=str, default=None, help="Unique identifier to load a trained model for testing."
    )
    parser.add_argument("--run-id", "-t", type=str, default="NAV", help="ID to distinguish different runs.")
    parser.add_argument(
        "--kwargs",
        "-k",
        type=str,
        default="",
        help="Dictionary type of other parameters. " "Format is like `a:1|b:2|c:test`",
    )
    parser.add_argument("--num-layers", "-l", type=int, default=3, help="Number of layers in every DNN.")
    parser.add_argument(
        "--polynomial-degrees",
        "-m",
        type=str,
        default="2|2",
        help="degrees of the trained encoding and computation function",
    )

    parser.add_argument("--num-inputs", "-n", type=int, default=3, help="Number of input matrices in dataset.")

    parser.add_argument(
        "--rand-seed", "-z", type=int, default=0, help="Random seed for generating training or test data."
    )

    parser.add_argument("--debug", action="store_true", help="Set when debugging on CPU machine.")

    args = parser.parse_args()

    N = int(args.matrix_dim)
    n_nodes = args.num_nodes
    n_layers = args.num_layers
    log_dir = str(args.log_dir).strip()
    save_dir = str(args.save_dir).strip()
    train_flag = bool(args.train)
    test_flag = bool(args.test)
    model_id = args.model_id
    if model_id is not None:
        model_id = model_id.strip()
    kwargs_str = str(args.kwargs).strip()
    rand_seed = args.rand_seed
    if rand_seed == 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rand_seed)

    degrees_str = args.polynomial_degrees
    num_inputs = args.num_inputs
    debug = args.debug
    run_id = args.run_id.strip()

    if debug:
        sep = "\\|"
    else:
        sep = "|"

    str_arr_parse = lambda str_arr: [s for s in str_arr.split(sep)]
    degrees = list(map(int, str_arr_parse(degrees_str)))
    recovery_threshold = np.prod(degrees) + 1

    kwargs = {}
    for s in kwargs_str.split(sep):
        tmp = s.split(":")
        if len(tmp) == 2:
            kwargs[tmp[0]] = tmp[1]

    
    # check for other params and set defaults
    extra_params = {
        "problem": "eigenvalue",
        "activation": "swish",
        "distribution": "uniform",
        "batch_size": 128,
        "steps_per_epoch": 128,
        "num_epochs": 100,
        "stopping_threshold": 1e-5,
        "learning_rate": 1e-3,
        "kernel_reg_factor": 0.0,
        "final_learning_rate": 1e-6,
        "decay_steps": 10,
        "validation_freq": 10,
        "eval_batch_size": 1024,
        "eval_num_reps": 1000,
    }

    for key in extra_params:
        try:
            if key in kwargs:
                if isinstance(extra_params[key], str):
                    extra_params[key] = str(kwargs[key])
                elif isinstance(extra_params[key], int):
                    extra_params[key] = int(kwargs[key])
                else:
                    extra_params[key] = float(kwargs[key])
        except e:
            print("Error occured in key %s"%key, " with value ", kwargs[key], " type ", type(kwargs[key]),  " and given type ", type(extra_params[key]))
            raise Exception(e)

    # check if we are using GPUs -> Keras should automatically pick available GPUs
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if extra_params["problem"] == "eigenvalue":
        output_dim = N
        extra_params["dist_param_1"] = -1.0
        extra_params["dist_param_2"] = 1.0
    elif extra_params["problem"] == "max_eigenvec":
        output_dim = N
        extra_params["dist_param_1"] = 0.0
        extra_params["dist_param_2"] = 1.0
    elif extra_params["problem"] == "matrix_exp":
        output_dim = N**2
        extra_params["dist_param_1"] = 0.0
        extra_params["dist_param_2"] = 1.0
    elif extra_params["problem"] == "matrix_det":
        output_dim = 1
        extra_params["dist_param_1"] = -2.0/N
        extra_params["dist_param_2"] = 2.0/N
    else:
        raise NotImplementedError(
            "Supported `problem`  are `eigenvalue`, `max_eigenvec`, `matrix_exp` and `matrix_det`." " `%s` is not supported." % extra_params["problem"]
        )

    model = create_model(
        N,
        output_dim,
        num_inputs,
        n_nodes,
        degrees,
        num_layers=n_layers,
        activation=extra_params["activation"]
    )

    print(model.summary())

    rng_tmp = np.random.default_rng()

    if model_id is not None:
        if debug:
            dir_path = os.path.join("./HPC_test/saved_models", model_id)
        else:
            dir_path = os.path.join(save_dir, model_id)

        model_save_filename = os.path.join(dir_path, "checkpoint")

        model.load_weights(model_save_filename)
    else:
        # unique model id
        model_id = "%s_%s_D%d_I%d_N%d_L%d_Pe%d_Ph%d_%04x" % (
            extra_params["problem"],
            run_id,
            N,
            num_inputs,
            n_nodes,
            n_layers,
            degrees[0],
            degrees[1],
            rng_tmp.integers(16**6),
        )
        if debug:
            dir_path = os.path.join("./HPC_test/saved_models", model_id)
        else:
            dir_path = os.path.join(save_dir, model_id)

        os.makedirs(dir_path, exist_ok=True)
        model_save_filename = os.path.join(dir_path, "checkpoint")

    def my_print(txt):
        print(txt)
        print_filename = os.path.join(dir_path, "program_output.out")

        if isinstance(txt, dict):
            txt = json.dumps(txt, sort_keys=False, indent=2)

        with open(print_filename, "a+") as f:
            f.write("%s\n" % txt)

    # my_print(model.summary())
    print("GPU name: ", gpus)
    print("Model identifier: %s" % model_id)
    my_print(vars(args))
    my_print(extra_params)
    my_print("Model identifier: %s" % model_id)

    my_print(
        "number of learnable params = %d" % np.sum([np.prod(tf.shape(a).numpy()) for a in model.trainable_variables]),
    )

    # save script to saveDir
    filename = os.path.basename(__file__)
    cmd = "cp %s %s/%s" %(filename, dir_path, filename)
    os.system(cmd)

    if debug:
        # Visualize the model
        model_visual_file = os.path.join(dir_path, "model_img.png")
        keras.utils.plot_model(model, to_file=model_visual_file, show_shapes=True)

    if train_flag:
        # use tensorboard for logging
        if debug:
            tb_logdir = os.path.join("./HPC_test/tb_logs/", model_id)
        else:
            tb_logdir = os.path.join(log_dir, model_id)

        my_print("TF log dir = %s" % tb_logdir)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_logdir, update_freq=1)

        my_print("Model save dir = %s" % model_save_filename)

        # learning rate decay
        steps_per_epoch = int(extra_params["steps_per_epoch"])
        initial_learning_rate = float(extra_params["learning_rate"])
        final_learning_rate = extra_params["final_learning_rate"]
        decay_steps = extra_params["decay_steps"]
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (
            decay_steps / extra_params["num_epochs"]
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps * steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True,
        )

        if extra_params["problem"] == "max_eigenvec":
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=custom_loss_reg, metrics=[custom_loss_nmse, custom_eigvec_norm])
        else:
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=keras.losses.MeanSquaredError(), metrics=[custom_loss_nmse])

        # model checkpoint callback
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=model_save_filename,
            save_weights_only=True,
            monitor="custom_loss_nmse",
            mode="min",
            save_only_best=True,
        )

        data_gen_params = {
            "rng": rng,
            "matrix_dim": N,
            "num_inputs": num_inputs,
            "batch_size": extra_params["batch_size"],
            "params": {
                "distribution": extra_params["distribution"],
                "dist_param_1": extra_params["dist_param_1"],
                "dist_param_2": extra_params["dist_param_2"],
                "problem": extra_params["problem"]
            }
        }

        datagen_obj = DataGenerator(data_gen_params, steps_per_epoch, recovery_threshold)

        training_history = model.fit(
            datagen_obj,
            epochs=int(extra_params["num_epochs"]),
            steps_per_epoch=steps_per_epoch,
            shuffle=False,
            verbose=1,
            callbacks=[tensorboard_callback, model_checkpoint_callback],
            validation_data=None,
        )

        my_print("Training done!")

    if test_flag:

        print("Starting Validation!")

        if model_id is None:
            raise Exception("The `model_id` is required when not training.")

        # load best weights
        model.load_weights(model_save_filename)


        rng = np.random.default_rng(rand_seed+1)

        data_gen_params = {
            "rng": rng,
            "matrix_dim": N,
            "num_inputs": num_inputs,
            "batch_size": extra_params["eval_batch_size"],
            "params": {
                "distribution": extra_params["distribution"],
                "dist_param_1": extra_params["dist_param_1"],
                "dist_param_2": extra_params["dist_param_2"],
                "problem": extra_params["problem"]
            },
        }

        num_reps = extra_params["eval_num_reps"]
        results = np.zeros((num_reps, 3))

        test_interpolation = True

        num_bins = 100
        if extra_params["problem"] == "matrix_det":
            n_hist_samples = 1
            random_eig_ids = [0]
        else:
            n_hist_samples = min(6, N)
            random_eig_ids = rng.choice(np.arange(N), n_hist_samples, replace=False)
        
        bins = np.zeros((n_hist_samples, num_bins + 1), dtype=float)
        hist = np.zeros((3, n_hist_samples, num_bins), dtype=float)
        interp_results = 0.0

        # Remember to match with the ones in convert_data_tf function
        betas = np.arange(1, num_inputs + 1) / num_inputs
        alphas = np.arange(1, recovery_threshold + 1) / (recovery_threshold + 1)

        for i in range(num_reps):
            X_data, y_data = random_bulk_data(**data_gen_params)

            interp_result = 0.0

            for j in range(n_hist_samples):
                if i == 0:
                    bins[j, :] = np.histogram_bin_edges(y_data[:, :, random_eig_ids[j]], bins=num_bins)
                hist_, _ = np.histogram(y_data[:, :, random_eig_ids[j]], bins=bins[j, :], density=False)
                hist[0, j, :] += hist_

            X_dnn, y_dnn = convert_data_tf(X_data, y_data, num_inputs, recovery_threshold)

            y_dnn = tf.cast(y_data, tf.float32)

            if test_interpolation:  # test lagrange interpolation

                X_dnn_val, y_dnn_val = convert_data_tf(X_data, y_data, num_inputs, recovery_threshold, for_training=False)

                y_dnn_val = tf.cast(y_dnn_val, tf.float32)

                computer_output = model(X_dnn_val).numpy()

                comp_out_val = np.reshape(computer_output, [extra_params["eval_batch_size"], recovery_threshold, -1])

                lang_out = lagrange_interpolator(alphas, betas, comp_out_val)

                est_y_dnn_interp = np.reshape(lang_out, [num_inputs * extra_params["eval_batch_size"], -1])

                interp_result = custom_loss_nmse_numpy(y_data, est_y_dnn_interp)

                results[i, 2] = interp_result

                interp_results += interp_result

                for j in range(n_hist_samples):
                    hist_, _ = np.histogram(est_y_dnn_interp[:, random_eig_ids[j]], bins=bins[j, :], density=False)
                    hist[2, j, :] += hist_

            tic = time.time()

            # decoder output
            est_y_dnn = model(X_dnn)

            toc = time.time()

            for j in range(n_hist_samples):
                hist_, _ = np.histogram(est_y_dnn[:, random_eig_ids[j]], bins=bins[j, :], density=False)
                hist[1, j, :] += hist_

            # est_y_dnn = model(X_dnn)

            test_results = custom_loss_nmse_numpy(y_data, est_y_dnn.numpy())

            results[i, 0] = toc - tic
            results[i, 1] = test_results

            my_print(
                "%d. \t DNN \t runtime : %.3es \t NMSE : %.4e \t interp_NMSE : %.4e "
                % (i, results[i, 0], results[i, 1], interp_result)
            )

        my_print("Time taken by DNN : %s secs" % (np.mean(results[:, 0])))
        my_print("Evaluation accuracy: %.3e" % np.mean(results[:, 1]))
        my_print("Interpolation Evaluation accuracy: %.3e" % (interp_results / num_reps))

        if debug:
            results_save_filename = os.path.join(
                "./HPC_test/saved_models", model_id, "results_%04x" % rng_tmp.integers(16**6)
            )
            hist_save_filename = os.path.join("./HPC_test/saved_models", model_id, "hist_%s" % (model_id))
        else:
            results_save_filename = os.path.join(
                save_dir, model_id, "results_%04x" % rng_tmp.integers(16**6)
            )
            hist_save_filename = os.path.join(save_dir, model_id, "hist_%s" % (model_id))

        np.save(results_save_filename, results)
        np.savez(hist_save_filename, bins=bins, hist=hist)

        print(model_id)
