from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, Flatten, BatchNormalization,MaxPool2D,multiply,add, Activation,Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras import backend as K
from keras.regularizers import l2
import numpy as np
from keras.losses import  binary_crossentropy, mae
import random
import keras
from keras.optimizers import Adam
import os
import tensorflow as tf
import scipy.io as sio
import time

#用于计算FLOPs
# tf.compat.v1.disable_eager_execution()
# print(tf.__version__)


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(3000)

#  导入SAR图像
# img_size = 24
# dele_size = 12

img_size = 16
dele_size = 8

pool_num1 =3
pool_num2 =5
pool_num3 =7

ps_epoch = 1
vae_epoch =1
vae_classifer_epoch = 20

valid_num = 256
img_wide = 20
img_high = 47
class_num = 4

data = sio.loadmat('input_data/Bridge/size16/X_ps.mat')     #素描图数据
x_ps = ( np.array(data.get('ps_block')).astype('float32')/255).T

data1 = sio.loadmat('input_data/Bridge/size16/X_train.mat')    #无监督训练数据
x_train = ( np.array(data1.get('train')).astype('float32') /255).T

data2 = sio.loadmat('input_data/Bridge/size16/X_valid.mat')     #标签数据
x_valid = ( np.array(data2.get('valid_img')).astype('float32')/255).T  #原图
x_valid_ps = ( np.array(data2.get('valid_ps')).astype('float32')/255).T   #素描图
valid_mask = ( np.array(data2.get('valid_mask')).astype('float32')/255).T   #掩膜
x_valid_mask = np.multiply(x_valid,valid_mask )  #原图点乘掩膜
x_valid_ps_mask = np.multiply(x_valid_ps,valid_mask )  #素描图点乘掩膜

data3 = sio.loadmat('input_data/Bridge/size16/X_test.mat')     #测试数据
x_test = ( np.array(data3.get('test_img')).astype('float32')/255).T
x_test_ps = ( np.array(data3.get('test_ps')).astype('float32')/255).T
test_mask = ( np.array(data3.get('test_mask')).astype('float32')/255).T
x_test_mask = np.multiply(x_test,test_mask)
x_test_ps_mask = np.multiply(x_test_ps,test_mask)

y_train1 = np.zeros(716)
y_train2 = np.ones(valid_num)
y_train3 = np.ones(valid_num)*2
y_train4 = np.ones(valid_num)*3

y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4))
shuffle_y = random.sample(np.arange(0,y_train.shape[0]).tolist(),y_train.shape[0])

x_valid = x_valid[shuffle_y,:]
y_valid = y_train[shuffle_y]
y_valid_cate = to_categorical(y_valid, num_classes=class_num)

x_train = x_train.reshape(-1, img_size, img_size, 1)
x_valid = x_valid.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)
ps_train = x_ps.reshape(-1, img_size, img_size, 1)

x_valid_ps = x_valid_ps.reshape(-1, img_size, img_size, 1)
x_test_ps = x_test_ps.reshape(-1, img_size, img_size, 1)

x_valid_mask = x_valid_mask.reshape(-1, img_size, img_size, 1)
x_test_mask = x_test_mask.reshape(-1, img_size, img_size, 1)

valid_mask = valid_mask.reshape(-1, img_size, img_size, 1)
x_valid_ps_mask = x_valid_ps_mask.reshape(-1, img_size, img_size, 1)

test_mask = test_mask.reshape(-1, img_size, img_size, 1)
x_test_ps_mask = x_test_ps_mask.reshape(-1, img_size, img_size, 1)

# network parameters
input_shape = (img_size, img_size, 1)
batch_size = 64
kernel_size = 3
filters = 32
latent_dim = 64
original_dim = img_size*img_size
intermediate_dim = 128


def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def g0_loss(z_mean, z_log_var):
    c = 0.2  #0.2-->60
    rho1 = 0.3  #0.3-->100
    a = 0.01
    b = 1.0
    e = 0.000001
    z_mean = z_mean + 0.000001

    q_log_p = K.sum((np.log(c) + c * np.log(rho1) + ((1.0 + c) * np.log(b + rho1) * (((rho1 + b) / z_mean) ** (-K.exp(z_log_var +e)))) -
                     ((1.0 + c) * np.log(a + rho1) * (((rho1 + a) / z_mean) ** (-K.exp(z_log_var+e)))) +
                     ((1.0 + c) * (((rho1 + a) / z_mean) ** (-K.exp(z_log_var+e)))) / (-K.exp(z_log_var+e)) -
                     ((1.0 + c) * (((rho1 + b) / z_mean) ** (-K.exp(z_log_var+e)))) / (-K.exp(z_log_var+e))), axis=-1)
    q_log_q = K.sum((z_log_var + K.exp(z_log_var) * K.log(z_mean) +
                     ((1.0 + K.exp(z_log_var)) * K.log(b + z_mean)) * (((b + z_mean) / z_mean) ** (-K.exp(z_log_var))) -
                     ((1.0 + K.exp(z_log_var)) * K.log(a + z_mean)) * (((a + z_mean) / z_mean) ** (-K.exp(z_log_var))) +
                     ((1.0 + K.exp(z_log_var)) / (-K.exp(z_log_var+e))) * (((a + z_mean) / z_mean) ** (-K.exp(z_log_var))) -
                     ((1.0 + K.exp(z_log_var)) / (-K.exp(z_log_var+e))) * (((b + z_mean) / z_mean) ** (-K.exp(z_log_var)))), axis=-1)
    G0_loss = K.mean(q_log_q - q_log_p)/original_dim
    return G0_loss


def residual_module(x, K, stride, chanDim, reduce=False, reg=1e-4, bnEps=2e-5,bnMom=0.9):
    shortcut = x
    bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

    bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

    bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

    if reduce:
        shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

    x = add([conv3, shortcut])
    return x



input_img = Input(shape=input_shape, name='encoder_input')
x = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(input_img)
x = MaxPool2D(pool_size=2,strides=2,padding='same')(x)
x = Conv2D(filters=64, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)
# x = Conv2D(filters=128, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)
x = MaxPool2D(pool_size=2,strides=2,padding='same')(x)
shape = K.int_shape(x)
x = Flatten()(x)
x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, activation='sigmoid')(x)
z_log_var = Dense(latent_dim, activation='sigmoid')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
x = Dense(intermediate_dim, activation='relu')(z)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)
# x = Conv2DTranspose(filters=128, kernel_size=kernel_size,activation='relu', strides=1, padding='same')(x)
x = Conv2DTranspose(filters=64, kernel_size=kernel_size,activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(filters=32, kernel_size=kernel_size,activation='relu', strides=2, padding='same')(x)
vae_output = Conv2DTranspose(filters=1,kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(x)

vae = Model(input_img, vae_output, name='vae')
vae.summary()
# get_flops_params()


# ps_input = Input(shape=input_shape)
# x = Conv2D(filters=32, kernel_size=7, activation='relu', strides=1, padding='same')(ps_input)
# x = MaxPool2D(pool_size=2,strides=2,padding='same')(x)
# x = Conv2D(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')(x)
# x = MaxPool2D(pool_size=2,strides=2,padding='same')(x)
# x = Conv2D(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(x)
# x = Conv2DTranspose(filters=64, kernel_size=3,activation='relu', strides=2, padding='same')(x)
# x = Conv2DTranspose(filters=32, kernel_size=3,activation='relu', strides=2, padding='same')(x)
# ps_outputs = Conv2DTranspose(filters=1,kernel_size=7, activation='sigmoid', strides=1,padding='same')(x)


ps_input = Input(shape=input_shape)
x = Conv2D(filters=32, kernel_size=7, activation='relu', strides=1, padding='same')(ps_input)
x = Conv2D(filters=32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
x = Conv2D(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')(x)
x = Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)

x = residual_module(x, 128, stride=1, chanDim=1, reduce=True, bnEps=2e-5, bnMom=0.9)
x = residual_module(x, 128, stride=2, chanDim=1, reduce=True, bnEps=2e-5, bnMom=0.9)
x = residual_module(x, 128, stride=1, chanDim=1, reduce=True, bnEps=2e-5, bnMom=0.9)

x = Conv2DTranspose(filters=128, kernel_size=kernel_size,activation='relu', strides=2, padding='same')(x)
x = Conv2D(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')(x)
x = Conv2DTranspose(filters=64, kernel_size=kernel_size,activation='relu', strides=2, padding='same')(x)
x = Conv2D(filters=32, kernel_size=3, activation='relu', strides=1, padding='same')(x)
x = Conv2DTranspose(filters=32, kernel_size=kernel_size,activation='relu', strides=2, padding='same')(x)
ps_outputs = Conv2DTranspose(filters=1,kernel_size=7, activation='sigmoid', strides=1,padding='same')(x)

sketcher = Model(ps_input, ps_outputs, name='sketcher')
# sketcher.summary()
# get_flops_params()


mask= Input(shape=input_shape)
class_input = Input(shape=input_shape)
vae_outputs = vae(class_input)
x = keras.layers.concatenate([class_input,vae_outputs],axis=1)
x = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)
x = MaxPool2D(pool_size=2,strides=2,padding='same')(x)
x = Conv2D(filters=64, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)
x = Conv2D(filters=128, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
c_output = Dense(class_num, activation='softmax')(x)

ps_vae_outputs = sketcher(vae(class_input))

vae_outputs_mask = multiply([vae_outputs,mask])
ps_vae_outputs_mask = multiply([ps_vae_outputs,mask])

vae_classifer = Model([mask,class_input], [c_output, vae_outputs_mask, ps_vae_outputs_mask], name='vae_classifer')
# vae_classifer.summary()
# get_flops_params()


#GaussVAE_Loss
# reconstruction_loss = binary_crossentropy(K.flatten(input_img), K.flatten(vae_output))
# reconstruction_loss *= original_dim
# kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
# kl_loss = -0.5*( K.sum(kl_loss, axis=-1))
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)


#G0VAE_Loss
reconstruction_loss = mae(K.flatten(input_img), K.flatten(vae_output))
reconstruction_loss *= img_size
G0_loss = g0_loss(z_mean,z_log_var)
vae_loss = K.mean( reconstruction_loss +  0.2*G0_loss)
vae.add_loss(vae_loss)


vae.compile(optimizer=Adam(lr=0.01, beta_1=0.5))

pp=[]
for i in range(50):
    x=vae.fit(x_train,x_train, epochs=vae_epoch, batch_size=batch_size,shuffle=True)
    pp.append(x.history['loss'][0])
print(pp)



sketcher.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])
sketcher.fit(x_train,ps_train,epochs=ps_epoch,batch_size=batch_size,shuffle=True)


vae_classifer.compile(optimizer='adam',loss=['categorical_crossentropy', 'mean_squared_error','mean_absolute_error'],loss_weights=[1,0.7,0.5],metrics=['categorical_accuracy'])
vae_classifer.fit([valid_mask,x_valid],[y_valid_cate, x_valid_mask, x_valid_ps_mask],epochs=vae_classifer_epoch,batch_size=batch_size,shuffle=True)

# for epoch in range(vae_classifer_epoch):
#     loss = vae_classifer.test_on_batch([valid_mask,x_valid],[y_valid_cate, x_valid_mask, x_valid_ps_mask])
#     pass

s= time.time()
acc_all = vae_classifer.predict([test_mask,x_test])
cur_time = (time.time()-s)
print(cur_time)

acc=acc_all[0]
acc_1 = acc
acc_2 = acc

print(np.argmax(acc,axis=1))

#pooling 处理
def pooling_label(label,img_high,img_wide,pool_num):
    label = np.reshape(label,(img_high,img_wide))
    for i in range(img_high-pool_num+1):
        for j in range(img_wide-pool_num+1):
            a = np.reshape(label[i:i+pool_num,j:j+pool_num],(1,-1))[0]
            label_count = np.bincount(a)
            label[i:i + 1, j:j + 1] = np.argmax(label_count)
            pass
        pass
    label = np.reshape(label,(1,-1))[0]
    return label

#软标签显示分类结果
def soft_multiclass(acc):
    acc1 = np.reshape(acc[:(img_high+1)*(img_wide+1),:],(-1,class_num))
    acc2 = np.reshape(acc[(img_high+1)*(img_wide+1):(img_high+1)*(img_wide+1)+(img_high+1)*(img_wide),:],(-1,class_num))
    acc3 = np.reshape(acc[(img_high+1)*(img_wide+1)+(img_high+1)*(img_wide):(img_high+1)*(img_wide+1)+(img_high+1)*(img_wide)+(img_high)*(img_wide+1),:],(-1,class_num))
    acc4 = np.reshape(acc[(img_high+1)*(img_wide+1)+(img_high+1)*(img_wide)+(img_high)*(img_wide+1):,:],(-1,class_num))

    I1 = np.ones(((img_high+1)*img_size,(img_wide+1)*img_size,class_num))
    I2 = np.ones(((img_high+1)*img_size,img_wide*img_size,class_num))
    I3 = np.ones((img_high*img_size,(img_wide+1)*img_size,class_num))
    I4 = np.ones((img_high*img_size,img_wide*img_size,class_num))

    k = 0
    for i in range(0,(img_high+1)*img_size,img_size):
        for j in range(0,(img_wide+1)*img_size,img_size):
            I1[i:i+ img_size, j:j + img_size,:]=I1[i:i+img_size,j:j+img_size,:]*acc1[k,:]
            k = k+1
            pass
        pass

    k = 0
    for i in range(0,(img_high+1)*img_size,img_size):
        for j in range(0,img_wide*img_size,img_size):
            I2[i:i+ img_size, j:j + img_size,:]=I2[i:i+img_size,j:j+img_size,:]*acc2[k,:]
            k = k+1
            pass
        pass

    k = 0
    for i in range(0,img_high*img_size,img_size):
        for j in range(0,(img_wide+1)*img_size,img_size):
            I3[i:i+ img_size, j:j + img_size,:]=I3[i:i+img_size,j:j+img_size,:]*acc3[k,:]
            k = k+1
            pass
        pass

    k = 0
    for i in range(0,img_high*img_size,img_size):
        for j in range(0,img_wide*img_size,img_size):
            I4[i:i+ img_size, j:j + img_size,:]=I4[i:i+img_size,j:j+img_size,:]*acc4[k,:]
            k = k+1
            pass
        pass

    I1=I1[dele_size:((img_high+1)*img_size-dele_size),dele_size:((img_wide+1)*img_size-dele_size),:]
    I2=I2[dele_size:((img_high+1)*img_size-dele_size),:,:]
    I3=I3[:,dele_size:((img_wide+1)*img_size-dele_size),:]

    I_label = I1+I2+I3+I4
    # I_label =  I1

    return I_label


I_0 = soft_multiclass(acc)
I_1 = soft_multiclass(acc_1)
I_2 = soft_multiclass(acc_2)

I_label = I_0+I_1+I_2
I = np.argmax(I_label, axis=2)

I = pooling_label(I.astype(np.int64),img_wide=img_wide*img_size, img_high=img_high*img_size,pool_num=pool_num1)
I = pooling_label(I.astype(np.int64),img_wide=img_wide*img_size, img_high=img_high*img_size,pool_num=pool_num2)
I = pooling_label(I.astype(np.int64),img_wide=img_wide*img_size, img_high=img_high*img_size,pool_num=pool_num3)
I = np.reshape(I,(img_high*img_size,img_wide*img_size))

def label_to_rgb(I):
    I_rgb = np.ones((img_high*img_size,img_wide*img_size,3))
    for i in range(img_high*img_size):
        for j in range(img_wide*img_size):
            if I[i,j] == 5:     # 深红
                I_rgb[i, j, 0] = 128
                I_rgb[i, j, 1] = 0
                I_rgb[i, j, 2] = 0
            elif I[i,j] == 1  : # 绿
                I_rgb[i, j, 0] = 0
                I_rgb[i, j, 1] = 131
                I_rgb[i, j, 2] = 74
            elif I[i, j] == 6:  # 蓝
                I_rgb[i, j, 0] = 0
                I_rgb[i, j, 1] = 0
                I_rgb[i, j, 2] = 255
            elif I[i, j] == 4:  # 黄
                I_rgb[i, j, 0] = 255
                I_rgb[i, j, 1] = 255
                I_rgb[i, j, 2] = 0
            elif I[i, j] == 0 :  # 红
                I_rgb[i, j, 0] = 183
                I_rgb[i, j, 1] = 0
                I_rgb[i, j, 2] = 255
            elif I[i, j] == 3:  # 浅黄
                I_rgb[i, j, 0] = 255
                I_rgb[i, j, 1] = 217
                I_rgb[i, j, 2] = 157
            elif I[i, j] == 2:  # 蓝
                I_rgb[i, j, 0] = 0
                I_rgb[i, j, 1] = 0
                I_rgb[i, j, 2] = 255

            pass
        pass
    I_rgb = I_rgb.astype(np.uint8)
    return I_rgb

I_rgb = label_to_rgb(I)
# I_rgb=I
plt.figure('soft_I')
plt.imshow(I_rgb)
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.imsave('soft_I.png',I_rgb)



#硬标签显示分类结果
label = np.argmax(acc,axis=1)
I1 = np.ones(((img_high+1)*img_size,(img_wide+1)*img_size))
I2 = np.ones(((img_high+1)*img_size,img_wide*img_size))
I3 = np.ones((img_high*img_size,(img_wide+1)*img_size))
I4 = np.ones((img_high*img_size,img_wide*img_size))

label = np.reshape(label,(1,-1))[0]
label1 = label[:(img_high+1)*(img_wide+1)]
label2 = label[(img_high+1)*(img_wide+1):(img_high+1)*(img_wide+1)+(img_high+1)*(img_wide)]
label3 = label[(img_high+1)*(img_wide+1)+(img_high+1)*(img_wide):(img_high+1)*(img_wide+1)+(img_high+1)*(img_wide)+(img_high)*(img_wide+1)]
label4 = label[(img_high+1)*(img_wide+1)+(img_high+1)*(img_wide)+(img_high)*(img_wide+1):]

k = 0
for i in range(0,(img_high+1)*img_size,img_size):
    for j in range(0,(img_wide+1)*img_size,img_size):
        I1[i:i+ img_size, j:j + img_size]=I1[i:i+img_size,j:j+img_size]*label1[k]
        k = k+1
        pass
    pass

k = 0
for i in range(0,(img_high+1)*img_size,img_size):
    for j in range(0,img_wide*img_size,img_size):
        I2[i:i+ img_size, j:j + img_size]=I2[i:i+img_size,j:j+img_size]*label2[k]
        k = k+1
        pass
    pass

k = 0
for i in range(0,img_high*img_size,img_size):
    for j in range(0,(img_wide+1)*img_size,img_size):
        I3[i:i+ img_size, j:j + img_size]=I3[i:i+img_size,j:j+img_size]*label3[k]
        k = k+1
        pass
    pass

k = 0
for i in range(0,img_high*img_size,img_size):
    for j in range(0,img_wide*img_size,img_size):
        I4[i:i+ img_size, j:j + img_size]=I4[i:i+img_size,j:j+img_size]*label4[k]
        k = k+1
        pass
    pass

I1=I1[dele_size:((img_high+1)*img_size-dele_size),dele_size:((img_wide+1)*img_size-dele_size)]
I2=I2[dele_size:((img_high+1)*img_size-dele_size),:]
I3=I3[:,dele_size:((img_wide+1)*img_size-dele_size)]

I = np.ones((img_high*img_size,img_wide*img_size))
for i in range(img_high*img_size):
    for j in range(img_wide*img_size):
        count = np.bincount([I1[i,j],I2[i,j],I3[i,j],I4[i,j]])
        I[i,j] = np.argmax(count)


I = pooling_label(I.astype(np.int64),img_wide=img_wide*img_size, img_high=img_high*img_size,pool_num=pool_num1)
I = pooling_label(I.astype(np.int64),img_wide=img_wide*img_size, img_high=img_high*img_size,pool_num=pool_num2)
I = pooling_label(I.astype(np.int64),img_wide=img_wide*img_size, img_high=img_high*img_size,pool_num=pool_num3)

I = np.reshape(I,(img_high*img_size,img_wide*img_size))
I_rgb_hard = label_to_rgb(I)
plt.figure('I')
plt.imshow(I_rgb_hard)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()
