import keras 
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, Input, Add, Activation, GlobalAveragePooling2D, BatchNormalization, UpSampling2D, Concatenate, Reshape, Flatten, Average
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras import optimizers, regularizers
import keras.backend as K
# from livelossplot.keras import PlotLossesCallback
from train_plot import PlotLearning
from functools import partial


class ResNet_v4:
    def __init__(self, epochs=250, batch_size=128, load_weights=True,level="32",transfer=False,x_train=None,y_train=None,x_val=None,y_val=None,loss_fn=None,istransfer=False):
        self.name               = 'resnet_'+level+'_nofusion_nodiversity_avg' #'resnet_cifar10' if dataset=='cifar10'  else 'resnet_mnist'
        self.model_filename     = 'models/'+self.name+'.h5'
        self.stack_n            = 5
        self.num_classes        = 10
        self.img_rows           = x_train.shape[1] 
        self.img_cols           = x_train.shape[2] 
        self.img_channels       = x_train.shape[3] 
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.iterations         = x_train.shape[0] // self.batch_size   # MNIST数据集应该是60000
        self.weight_decay       = 0.0001
        self.log_filepath       = r'logs/v4/'
        self._model             = None  # 防止后续出现未定义的情况
        self.param_count        = None
        self.x_train,self.y_train    = x_train,y_train
        self.x_val,self.y_val      = x_val,y_val
        self.loss_fn            = None # loss_fn  if loss_fn is not None else 'categorical_crossentropy'  
        self.transfer            = istransfer
        
        if load_weights:
            try:
                self._model = load_model(self.model_filename)  # _model表示私有变量，保存了.h5加载过来的参数，load_model库函数详见models.py
                self.param_count = self._model.count_params()                
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)


    def Entropy(self, input):
    #input shape is batch_size X num_class
        return tf.reduce_sum(-tf.multiply(input, tf.log(input + 1e-20)), axis=-1)

    def Ensemble_Entropy(self, p3, p2, p1):
        y_p_all = 0
        y_p_all += p3
        y_p_all += p2
        y_p_all += p1
        Ensemble = self.Entropy(y_p_all / 3)
        return Ensemble


    def log_det(self, p3, p2, p1, y_true_one):
        zero = tf.constant(0, dtype=tf.float32)
        det_offset = 1e-6
        num_model = 3
        y_true = []
        y_true.append(y_true_one)
        y_true.append(y_true_one)
        y_true.append(y_true_one)
        y_true = K.concatenate(y_true)
        y_true = K.reshape(y_true, (-1, 10, 3))
        y_pred = []
        y_pred.append(p3)
        y_pred.append(p2)
        y_pred.append(p1)
        y_pred = K.concatenate(y_pred)
        y_pred = K.reshape(y_pred, (-1, 10, 3))
        bool_R_y_true = tf.not_equal(tf.ones_like(y_true) - y_true, zero) # batch_size X (num_class X num_models), 2-D
        mask_non_y_pred = tf.boolean_mask(y_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D
        mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, self.num_classes-1]) # batch_size X num_model X (num_class-1), 3-D
        mask_non_y_pred = mask_non_y_pred / tf.norm(mask_non_y_pred, axis=2, keepdims=True) # batch_size X num_model X (num_class-1), 3-D
        matrix = tf.matmul(mask_non_y_pred, tf.transpose(mask_non_y_pred, perm=[0, 2, 1])) # batch_size X num_model X num_model, 3-D
        all_log_det = tf.linalg.logdet(matrix+det_offset*tf.expand_dims(tf.eye(num_model),0)) # batch_size X 1, 1-D
        return all_log_det


    def custome_loss(self, y_true, y_pred, p3, p2, p1):
        lamda = 1
        log_det_lamda = 0.5
        ce_final = K.categorical_crossentropy(y_true, y_pred)
        ce_fusion = 0
        ce_fusion += K.categorical_crossentropy(y_true, p3)
        ce_fusion += K.categorical_crossentropy(y_true, p2)
        ce_fusion += K.categorical_crossentropy(y_true, p1)
        ee = self.Ensemble_Entropy(p3, p2, p1)
        log_dets = self.log_det(p3, p2, p1, y_true)
        return ce_final + ce_fusion - lamda * ee - log_det_lamda * log_dets

    
    def construct(self):
        # build network 
        img_input = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
        output, _, _, _ = self.residual_network(img_input,self.num_classes,self.stack_n)
        self._model = Model(img_input, output)
        self.loss_fn = 'categorical_crossentropy'
        # self.loss_fn = partial(self.custome_loss, p3=p3, p2=p2, p1=p1)
        # self.loss_fn.__name__ = "EE_ADP"
        # self._model.summary()

        

    def scheduler(self, epoch):
        lr = 1e-3
        if epoch > 220:
            lr *= 1e-3
        elif epoch > 150:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def residual_network(self, img_input,classes_num=10,stack_n=5):  # ResNet参数，但不包括weights
         # 一个残差块！！！ 从模型图片来看就是从一个Add到另一个Add之间的部分
        def residual_block(intput, out_channel,increase=False, stage=0, block=0):
            if increase:
                stride = (2,2)
            else:
                stride = (1,1)

            pre_bn   = BatchNormalization()(intput)  # （input）为这一层的输入
            pre_relu = Activation('relu')(pre_bn)

            conv_1 = Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same',
                            kernel_initializer="he_normal", # he_normal——he正态分布初始化
                            kernel_regularizer=regularizers.l2(self.weight_decay))(pre_relu)
            
            # 记得改回去 —— 修改残差块
            bn_1   = BatchNormalization()(conv_1)
            relu1  = Activation('relu')(bn_1)
            conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(relu1)
            if increase:  # 印证了结构图中有的会在右侧线路中多一个卷积层
                projection = Conv2D(out_channel,
                                    kernel_size=(1,1),
                                    strides=(2,2),
                                    padding='same',
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=regularizers.l2(self.weight_decay))(intput)
                block = Add(name='block_'+str(stage)+'_'+str(block))([conv_2, projection])
            else:
                block = Add(name='block_'+str(stage)+'_'+str(block))([intput,conv_2])       # 残差的概念：块中的最后一个输出和块中的第一个输入要相加
            return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32
        # input: 32x32x3 output: 32x32x16
        x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)

        # input: 32x32x16 output: 32x32x16
        for i in range(stack_n):
            x = residual_block(x,16,False, stage=1, block=i)   # 16就是channel！与输入维度无关，最终结果直接由输出维度决定

        exit1 = BatchNormalization()(x)
        exit1 = Activation('relu')(exit1)

        x = residual_block(x,32,True, stage=2, block=0)
        for i in range(1,stack_n):
            x = residual_block(x,32,False, stage=2, block=i)

        exit2 = BatchNormalization()(x)
        exit2 = Activation('relu')(exit2)
        
        # input: 16x16x32 output: 8x8x64
        x = residual_block(x,64,True, stage=3, block=0)
        for i in range(1,stack_n):
            x = residual_block(x,64,False, stage=3, block=i)

        exit3 = BatchNormalization()(x)
        exit3 = Activation('relu')(exit3)

        p3 = Conv2D(64, kernel_size=(1,1))(exit3)
        p2 = Conv2D(64, kernel_size=(1,1))(exit2)
        p1 = Conv2D(64, kernel_size=(1,1))(exit1)
        # p2 = Add()([
        #     UpSampling2D(size=(2,2))(p3),
        #     Conv2D(64, kernel_size=(1,1))(exit2)
        # ])
        # p1 = Add()([
        #     UpSampling2D(size=(2,2))(p2),
        #     Conv2D(64, kernel_size=(1,1))(exit1)
        # ])

        # 16*16*128
        p1 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(p1)
        # 8*8*128
        p1 = Conv2D(128, kernel_size=(1, 1), strides=(2, 2), padding='same')(p1)

        p1 = GlobalAveragePooling2D()(p1)

        p1 = Dense(classes_num,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(p1)

        # 8*8*128
        p2 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(p2)

        p2 = GlobalAveragePooling2D()(p2)

        p2 = Dense(classes_num,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(p2)

        # 8*8*128
        p3 = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(p3)

        p3 = GlobalAveragePooling2D()(p3)

        p3 = Dense(classes_num,
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(p3)

        p1 = Activation('softmax')(p1)  
        p2 = Activation('softmax')(p2)
        p3 = Activation('softmax')(p3)
        
        out_inner = []
        out_inner.append(p3)
        out_inner.append(p2)
        out_inner.append(p1)
        out = Average()(out_inner)

        return out, p3, p2, p1

    def train(self):          
        if not self.transfer:
            self.construct()
        
        if self.x_val is not None and self.y_val is not None:
            validation_data=(self.x_val, self.y_val)
        else:
            validation_data=None

        # set optimizer
        # sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True) # momentum-动量法,nesterov-NAG
        adam = optimizers.Adam()
        
        self._model.compile(loss=self.loss_fn, optimizer=adam, metrics=['accuracy'])
        self._model.summary()

        # set callback
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(self.scheduler)
        checkpoint = ModelCheckpoint('models/'+self.name+'_weight_best'+'.h5', # 这里保存了model
                monitor='val_acc', verbose=0, save_best_only= True, mode='auto', save_weights_only=True)
        plot_callback = PlotLearning()
        cbks = [ tb_cb, checkpoint, plot_callback, change_lr]

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=0.125,
                                    height_shift_range=0.125,
                                    fill_mode='constant',cval=0.)

        datagen.fit(self.x_train)

        # def generate_3out(generator, x, y, batch_size):
        #     gen = generator.flow(x, y, batch_size=batch_size)
        #     while True:
        #         (x1, y1)  = gen.next()
        #         # y2 = y1.copy()
        #         # y3 = y1.copy()
        #         yield (x1, [y1, y1 ,y1])


        # start training
        self._model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                            steps_per_epoch=self.iterations,
                            epochs=self.epochs,
                            callbacks=cbks,
                            validation_data=validation_data)
        # self._model.save('models/'+self.name+'_3out'+'.h5')
        self.param_count = self._model.count_params()


    def accuracy(self):
        return self._model.evaluate(self.x_val, self.y_val, verbose=0)
