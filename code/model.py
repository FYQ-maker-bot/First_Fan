from tensorflow.python.keras.layers import Input, merge, Flatten
import csv
from keras.utils.vis_utils import plot_model
import numpy as np
import keras.utils.np_utils as kutils
from keras.optimizers import adam_v2
from keras.layers import Conv1D,Conv2D, MaxPooling2D, MaxPooling1D, UpSampling1D, Cropping1D
from keras.regularizers import l2, l1
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add, Lambda, Concatenate
from keras import layers, Model
from sklearn import metrics
from keras.callbacks import EarlyStopping
from keras.layers import Reshape
from keras import backend as K
from sklearn.preprocessing import normalize
from keras.losses import MSE, KLDivergence, BCE


def getMatrixLabelh(positive_position_file_name, window_size=50, empty_aa='*'):

    rawseq = []
    all_label = []
    length = []
    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
            a = window_size - len(row[1])
            sseq = row[1]
            rawseq.append(sseq)
            b = len(row[1])
            length.append(b)
            all_label.append(int(row[0]))
        targetY = kutils.to_categorical(all_label)

        ONE_HOT_SIZE = 20
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19

        Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in rawseq:
            AANo = 0
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo + 1
            samplenumber = samplenumber + 1

    return Matr, targetY, rawseq, length,all_label


def vae_loss(recon_x, x):
    # 重构损失
    reconstruction_loss = MSE(recon_x, x)
    loss = K.mean(reconstruction_loss)

    return loss


def ae_e(x, n_output, weight_decay2):

    encoded = Dense(1024, activation='relu', kernel_initializer=init_form,
                    kernel_regularizer=l2(weight_decay2),
                    bias_regularizer=l2(weight_decay2))(x)
    encoded = Dense(512, activation='relu', kernel_initializer=init_form,
                    kernel_regularizer=l2(weight_decay2),
                    bias_regularizer=l2(weight_decay2))(encoded)
    encoder_output = Dense(n_output, activation='relu', kernel_initializer=init_form,
                           kernel_regularizer=l2(weight_decay2),
                           bias_regularizer=l2(weight_decay2))(encoded)
    return encoder_output


def ae_d(x, n_intput, weight_decay2):

    decoded = Dense(512, activation='relu', kernel_initializer=init_form,
                    kernel_regularizer=l2(weight_decay2),
                    bias_regularizer=l2(weight_decay2))(x)
    decoded = Dense(1024, activation='relu', kernel_initializer=init_form,
                    kernel_regularizer=l2(weight_decay2),
                    bias_regularizer=l2(weight_decay2))(decoded)
    decoded_output = Dense(n_intput, activation='sigmoid', kernel_initializer=init_form,
                           kernel_regularizer=l2(weight_decay2),
                           bias_regularizer=l2(weight_decay2))(decoded)
    return decoded_output


def cae_e(x, kernel_size=3, weight_decay=0.1):
    x1 = Conv1D(16, kernel_size, activation='relu', padding='same', kernel_initializer=init_form,
                kernel_regularizer=l2(weight_decay))(x)
    x2 = MaxPooling1D(2, padding='same')(x1)
    x3 = Conv1D(4, kernel_size, activation='relu', padding='same', kernel_initializer=init_form,
                kernel_regularizer=l2(weight_decay))(x2)
    encoded = MaxPooling1D(2, padding='same')(x3)

    return encoded


def cae_d(x, n_input, c, kernel_size=3, weight_decay=0.1):
    y1 = Conv1D(4, kernel_size, activation='relu', padding='same', kernel_initializer=init_form,
                kernel_regularizer=l2(weight_decay))(x)
    y1 = UpSampling1D(2)(y1)
    y2 = Conv1D(16, kernel_size, activation='relu', padding='same', kernel_initializer=init_form,
                kernel_regularizer=l2(weight_decay))(y1)
    y3 = UpSampling1D(2)(y2)
    decoded = Conv1D(n_input, kernel_size, activation='relu', padding='same', kernel_initializer=init_form,
                     kernel_regularizer=l2(weight_decay))(y3)

    if c:
        decoded = Cropping1D(cropping=(0, 1))(decoded)

    return decoded


def Phos1(nb_classes, img_dim1, img_dim2, init_form, weight_decay):

    main_input = Input(shape=img_dim1)
    input2 = Input(shape=img_dim2)
    c1 = Reshape((100, 20))(main_input)
    c2 = Reshape((91, 17))(input2)

    z_a1 = ae_e(main_input, 100, 0.001)
    z_a2 = ae_e(input2, 92, 0.001)

    z_c1 = cae_e(c1)
    z_c2 = cae_e(c2)

    z_a11 = Reshape((25, 4))(z_a1)
    z_a22 = Reshape((23, 4))(z_a2)
    z_c11 = Reshape((100,))(z_c1)
    z_c22 = Reshape((92,))(z_c2)

    zza_1 = z_a1 + z_c11
    zzc_1 = z_c1 + z_a11
    zza_2 = z_a2 + z_c22
    zzc_2 = z_c2 + z_a22

    ae_out1 = ae_d(zza_1, 2000, 0.001)
    ae_out2 = ae_d(zza_2, 1547, 0.001)
    cae_out1 = cae_d(zzc_1, 20, 0)
    cae_out2 = cae_d(zzc_2, 17, 1)
    x1 = Conv1D(64, 1, padding="same", kernel_regularizer=l2(weight_decay))(zzc_1)
    x2 = Conv1D(64, 1, padding="same", kernel_regularizer=l2(weight_decay))(zzc_2)

    x = concatenate([x1, x2], axis=-2, name='contact_multi_seq')
    x = Flatten()(x)
    x = Dense(nb_classes,
              name='Dense_softmax',
              activation='softmax',
              kernel_initializer=init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    phos_model = Model(inputs=[main_input, input2],
                       outputs=[x], name="multi-DenseNet")
    phos_model.add_loss(vae_loss(ae_out1, main_input))
    phos_model.add_loss(vae_loss(ae_out2, input2))
    phos_model.add_loss(vae_loss(cae_out1, c1))
    phos_model.add_loss(vae_loss(cae_out2, c2))
    return phos_model


num_train = 12824
num_val = 3208
train_file_name = ""
win1 = 100
X1, T, raw, length, y_ture1 = getMatrixLabelh(train_file_name, win1)
aaa = np.zeros(shape=(num_train, 100, 20))
bbb = np.zeros(shape=(num_train, 1547))

aaa2 = np.zeros(shape=(num_val, 100, 20))
bbb2 = np.zeros(shape=(num_val, 1547))

y_ture = np.zeros(shape=(num_train,))
aaa[:] = X1[:]
X2 = np.load("")

bbb[:] = X2[:]
ddd = np.zeros(shape=(num_train, 2))
ddd2 = np.zeros(shape=(num_val, 2))

ddd[:] = T[:]
y_ture[:] = y_ture1[:]

aaa = aaa.reshape((num_train, 2000))
bbb = normalize(bbb, axis=1, norm="l2")

val_file_name = ""
win1 = 100
X_text, T3, raw3, length3, y_ture3 = getMatrixLabelh(val_file_name, win1)
X2_text = np.load("")
aaa2[:] = X_text[:]
bbb2[:] = X2_text[:]
ddd2[:] = T3[:]
aaa2 = aaa2.reshape((num_val, 2000))
bbb2 = normalize(bbb2, axis=1, norm="l2")

img_dim1 = aaa.shape[1:]

img_dim2 = bbb.shape[1:]

init_form = 'RandomUniform'
learning_rate = 0.00001
filter_size_ori = 1
dense_number = 36
weight_decay = 0.001
nb_batch_size = 256
nb_classes = 2
nb_epoch = 100
class_weights = {0: 16032/(2*12024), 1: 16032/(2*4008)}

model1 = Phos1(nb_classes, img_dim1, img_dim2, init_form, weight_decay)

print(model1.summary())

plot_model(model1, to_file='modle.png', show_shapes=True, show_layer_names=True)

opt = adam_v2.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model1.compile(loss={'Dense_softmax': 'binary_crossentropy'},
               optimizer=opt, metrics=['accuracy'])

earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100,
                          mode='max', verbose=1, restore_best_weights=True)

history = model1.fit([aaa, bbb], y=ddd, batch_size=nb_batch_size,
                     validation_data=([aaa2, bbb2], [ddd2]),
                     epochs=nb_epoch, class_weight=class_weights, shuffle=True, verbose=1)

model1.save('', overwrite=True)

