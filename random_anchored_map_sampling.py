import numpy as np
from torch.autograd import Variable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, PReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from util import do_inverse_norm, ReduceLROnPlateau

# Set default type for compatibility between CPU/GPU
tf.keras.backend.set_floatx('float32')
#tf.config.run_functions_eagerly(True)

class deep_ensemble():

    def __init__(self, inF, outF, H, lr=1e-4, Nmodels=5, problem='regression', epsilon=1e-3):
        self.inF= inF
        self.outF = outF
        self.H = H
        self.problem = problem
        self.lr = lr
        self.model_fn = self.base_model_regression
        self.train_step = self.train_step_regression
        self.pred = self.pred_regression
        self.loss_fn = self.loss_reg
        if(problem=='classification'):
            self.model_fn = self.base_model_classification
            self.train_step = self.train_step_classification
            self.pred = self.pred_classification
            self.loss_fn = self.loss_class

        self.models = []
        self.Nmodels = Nmodels
        self.optimizers = []
        for i in range(Nmodels):
            self.models.append(self.model_fn())
            opt=tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999)
            grad_vars = self.models[i].trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            opt.apply_gradients(zip(zero_grads, grad_vars)) 
            self.optimizers.append(opt)
        # get init kernel weights
        self.init_wghts = []
        self.lambda_anchor = []
        # variance of noise in data (aleatoric uncertainty)
        # for present case should be small
        self.epsilon = epsilon
        for i in range(Nmodels):
            wght_list = []
            for wghts in self.models[i].trainable_weights:
                if('kernel' in wghts.name):
                    wght_list.append(wghts.numpy())
                    if(i==0):
                        self.lambda_anchor.append(epsilon/(2/(sum(wght_list[-1].shape))))
            self.init_wghts.append(wght_list)

        return


    def base_model_regression(self):

        init = tf.keras.initializers.glorot_normal()
        #def _kernel_init(scale=1.0, seed=None):
        #    """He normal initializer with scale."""
        #    scale = 2. * scale
        #    return tf.keras.initializers.VarianceScaling(
        #    scale=scale, mode='fan_in', distribution="truncated_normal", seed=seed)
        #init = _kernel_init(scale=0.1)
        inputs = Input(shape=(self.inF,))
        x = Dense(self.H, kernel_initializer=init)(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dense(self.H, kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dense(self.H, kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dense(self.outF, kernel_initializer=init)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        model.build(input_shape=(self.inF))
        return model

    def base_model_classification(self):

        init = tf.keras.initializers.glorot_normal()
        
        inputs = Input(shape=(self.inF,))
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(inputs)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dense(self.outF, activation='sigmoid', kernel_initializer=init)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model


    def _generate_ensembles(self, N):
        self.models = []
        for i in range(N):
            self.models.append(self.model_fn())

        return

    def loss_reg(self, model, xtrain, ytrain, indx, training):
        #mse_loss = tf.keras.losses.MSE(ytrain, model(xtrain, training=training))
        ypred = model(xtrain, training=training)
        mse_loss = tf.reduce_mean(tf.math.square(ytrain - ypred))
        anchor_loss = 0
        i=0
        for wghts in model.trainable_weights:
            if('kernel' in wghts.name):
                anchor_loss = anchor_loss + tf.reduce_mean(tf.math.square(self.init_wghts[indx][i] - wghts))*self.lambda_anchor[i]
                i=i+1

        return  mse_loss + anchor_loss, mse_loss, anchor_loss


    @tf.function
    def loss_class(self, model, xtrain, ytrain, indx, training):
        # 
        ypred = model(xtrain, training=training)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        entrp_loss = loss_fn(ytrain, ypred)
        anchor_loss=0
        i=0
        for wghts in model.trainable_weights:
            if('kernel' in wghts.name):
                anchor_loss = anchor_loss + tf.reduce_mean(tf.math.square(self.init_wghts[indx][i] - wghts))*self.lambda_anchor[i]
                i=i+1
        return entrp_loss + anchor_loss, entrp_loss, anchor_loss
    @tf.function
    def train_step_regression(self, model, xtrain, ytrain, optimizer, indx):
        with tf.GradientTape() as tape:
            total_loss, mse_loss, anchor_loss  = self.loss_fn(model,xtrain, ytrain, indx, True)
            #loss = tf.keras.losses.mse(ytrain,model(xtrain))
        grad = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return total_loss, mse_loss, anchor_loss
    @tf.function
    def train_step_classification(self, model, xtrain, ytrain, optimizer, indx):
        with tf.GradientTape() as tape:
            total_loss, entrp_loss, anchor_loss = self.loss_fn(model, xtrain, ytrain, indx,  True)
        grad = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return total_loss, entrp_loss, anchor_loss

    def train(self, model, batch_size, epochs, xtrain, ytrain, indx, validation_data=None):
        train_dataset = tf.data.Dataset.from_tensor_slices((xtrain.astype(np.float32), ytrain.astype(np.float32)))
        train_dataset = train_dataset.shuffle(buffer_size=xtrain.shape[0], reshuffle_each_iteration=True).batch(batch_size)
        #self.optimizer = tf.keras.optimizers.RMSprop(self.lr)
        red_lr = ReduceLROnPlateau(self.optimizers[indx], 0.8, 10, 1e-5)

        if(validation_data):
            validation_data[0] = validation_data[0].astype(np.float32)
            validation_data[1] = validation_data[1].astype(np.float32)

        train_loss = []
        valid_loss = []
        mse_loss = []
        anchor_loss = []
        epoch_loss_avg = tf.keras.metrics.Mean()
        mse_loss_avg = tf.keras.metrics.Mean()
        anchor_loss_avg = tf.keras.metrics.Mean()

        for i in range(epochs):
            epoch_loss_avg.reset_states()
            mse_loss_avg.reset_states()
            anchor_loss_avg.reset_states()
            for x, y in train_dataset:
                total_loss, mse, anchor = self.train_step(model, x, y, self.optimizers[indx], indx)
                epoch_loss_avg.update_state(total_loss)
                mse_loss_avg.update_state(mse)
                anchor_loss_avg.update_state(anchor)
            train_loss.append(epoch_loss_avg.result().numpy())
            mse_loss.append(mse_loss_avg.result().numpy())
            anchor_loss.append(anchor_loss_avg.result().numpy())
            red_lr.on_epoch_end(train_loss[-1], i)

            if(validation_data):
                valid_loss.append(np.mean(self.loss_fn(model, validation_data[0], validation_data[1], False).numpy()))
                print("Step {} loss {} valid_loss {}".format(i, epoch_loss_avg.result(), valid_loss[i]))
            else:
                print("Step {} loss {}".format(i, epoch_loss_avg.result()))
        
        return [train_loss, mse_loss, anchor_loss], valid_loss

    def train_ensemble(self, xtrain, ytrain, epochs, batch_size, validation_data=None):
        #self.models = []
        hist = []
        hist_valid = []
        for i in range(self.Nmodels):
            #self.models.append(self.model_fn())
            h1, h2 = self.train(self.models[i], batch_size, epochs, xtrain, ytrain, i, validation_data)
            hist.append(h1)
            hist_valid.append(h2)
        return hist, hist_valid

    def pred_regression(self, xdata, y, norm):
        pred_all = np.zeros([len(self.models),xdata.shape[0], self.outF])
        for i in range(len(self.models)):
            mu = self.models[i](xdata)
            pred_all[i,:,0:self.outF]  = mu

        pred_all = do_inverse_norm(y, pred_all, norm)
        pred_mean = np.mean(pred_all[:,:,0:self.outF],axis=0)
        pred_std = np.sqrt(np.abs(np.mean(pred_all[:,:,0:self.outF]**2, axis=0) - pred_mean**2 ))

        return pred_all, pred_mean, pred_std

    def pred_classification(self, xdata, y, norm):
        pred_all = np.zeros([len(self.models),xdata.shape[0], self.outF])
        for i in range(len(self.models)):
            pred_all[i,:,:] = self.models[i](xdata)

        pred_all = do_inverse_norm(y, pred_all, norm)
        # Mean, std as in paper
        pred_mean = np.mean(pred_all,axis=0)
        pred_std = np.std(pred_all,axis=0)
        return pred_all, pred_mean, pred_std

    def predict(self, xdata, y, norm):
        pred_all, pred_mean, pred_std = self.pred(xdata, y, norm)
        return pred_all, pred_mean, pred_std
