import tensorflow.keras
import os
import numpy as np
'''
#Get a models config:

model = tensorflow.keras.models.load_model("C:\\Users\Aki\saved_model\keras_model.h5")
print("Full model:")
model.summary()
print("layers[0]:")
model.layers[0].summary()
print("layers[0][0]:")
model.layers[0].layers[0].summary()
#print("layers[0][1]:")
#model.layers[0].layers[1].summary()
print("layers[1]:")
model.layers[1].summary()




config = model.layers[1].get_config()
for item in config:
    for subitem in config[item]:
        print(subitem)
        print("########################################################################")
'''
#model = tensorflow.keras.models.Sequential()
initializer = tensorflow.keras.initializers.VarianceScaling(scale=1, mode='fan_avg', distribution='uniform')

input=tensorflow.keras.Input(shape=(224,224,3), name="input_1")
layer=tensorflow.keras.layers.ZeroPadding2D(padding=((0,1),(0,1)), data_format= 'channels_last', name="Conv_1_pad")(input)
layer = tensorflow.keras.layers.Conv2D(16, kernel_size=(3,3), strides=(2,2), data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="Conv1")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="bn_Conv1")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="Conv_1_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="expanded_conv_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="expanded_conv_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="expanded_conv_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(8, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="expanded_conv_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="expanded_conv_project_BN")(layer)
layer = tensorflow.keras.layers.Conv2D(48, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_1_expand")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_1_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_1_expand_relu")(layer)
layer = tensorflow.keras.layers.ZeroPadding2D(padding=((0,1),(0,1)), data_format= 'channels_last', name="block_1_pad")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (2, 2), padding = "valid", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_1_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_1_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_1_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(8, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_1_project")(layer)
adding1 = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_1_project_BN")(layer)

layer = tensorflow.keras.layers.Conv2D(48, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_2_expand")(adding1)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_2_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_2_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_2_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_2_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_2_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(8, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_2_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_2_project_BN")(layer)

layer = tensorflow.keras.layers.Add(name = "block_2_add")([adding1, layer])

layer = tensorflow.keras.layers.Conv2D(48, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_3_expand")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_3_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_3_expand_relu")(layer)
layer = tensorflow.keras.layers.ZeroPadding2D(padding=((0,1),(0,1)), data_format= 'channels_last', name="block_3_pad")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (2, 2), padding = "valid", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_3_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_3_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_3_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(16, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_3_project")(layer)
adding3 = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_3_project_BN")(layer)

layer = tensorflow.keras.layers.Conv2D(96, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_4_expand")(adding3)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_4_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_4_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_4_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_4_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_4_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(16, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_4_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_4_project_BN")(layer)

adding4 = tensorflow.keras.layers.Add(name = "block_4_add")([adding3, layer])

layer = tensorflow.keras.layers.Conv2D(96, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_5_expand")(adding4)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_5_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_5_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_5_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block__depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_5_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(16, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_5_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_5_project_BN")(layer)

adding5 = tensorflow.keras.layers.Add(name = "block_5_add")([adding4, layer])

layer = tensorflow.keras.layers.Conv2D(96, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_6_expand")(adding5)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_6_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_6_expand_relu")(layer)
layer = tensorflow.keras.layers.ZeroPadding2D(padding=((0,1),(0,1)), data_format= 'channels_last', name="block_6_pad")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (2, 2), padding = "valid", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_6_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_6_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_6_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(24, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_6_project")(layer)
adding6 = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_6_project_BN")(layer)

layer = tensorflow.keras.layers.Conv2D(144, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_7_expand")(adding6)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_7_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_7_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_7_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_7_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_7_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(24, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_7_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_7_project_BN")(layer)

adding7 = tensorflow.keras.layers.Add(name = "block_7_add")([adding6, layer])

layer = tensorflow.keras.layers.Conv2D(144, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_8_expand")(adding7)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_8_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_8_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_8_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_8_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_8_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(24, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_8_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_8_project_BN")(layer)

adding8 = tensorflow.keras.layers.Add(name = "block_8_add")([adding7, layer])

layer = tensorflow.keras.layers.Conv2D(144, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_9_expand")(adding8)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_9_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_9_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_9_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_9_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_9_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(24, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_9_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_9_project_BN")(layer)

adding9 = tensorflow.keras.layers.Add(name = "block_9_add")([adding8, layer])

layer = tensorflow.keras.layers.Conv2D(144, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_10_expand")(adding9)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_10_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_10_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_10_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_10_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_10_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(32, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_10_project")(layer)
adding10 = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_10_project_BN")(layer)

layer = tensorflow.keras.layers.Conv2D(192, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_11_expand")(adding10)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_11_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_11_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_11_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_11_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_11_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(32, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_11_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_11_project_BN")(layer)

adding11 = tensorflow.keras.layers.Add(name = "block_11_add")([adding10, layer])

layer = tensorflow.keras.layers.Conv2D(192, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_12_expand")(adding11)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_12_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_12_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_12_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_12_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_12_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(32, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_12_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_12_project_BN")(layer)

adding12 = tensorflow.keras.layers.Add(name = "block_12_add")([adding11, layer])

layer = tensorflow.keras.layers.Conv2D(192, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_13_expand")(adding12)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_13_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_13_expand_relu")(layer)
layer = tensorflow.keras.layers.ZeroPadding2D(padding=((0,1),(0,1)), data_format= 'channels_last', name="block_13_pad")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (2, 2), padding = "valid", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_13_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_13_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_13_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(56, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_13_project")(layer)
adding13 = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_13_project_BN")(layer)

layer = tensorflow.keras.layers.Conv2D(336, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_14_expand")(adding13)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_14_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_14_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_14_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_14_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_14_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(56, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_14_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_14_project_BN")(layer)

adding14 = tensorflow.keras.layers.Add(name = "block_14_add")([adding13, layer])

layer = tensorflow.keras.layers.Conv2D(336, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_15_expand")(adding14)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_15_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_15_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_15_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_15_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_15_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(56, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_15_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_15_project_BN")(layer)

adding15 = tensorflow.keras.layers.Add(name = "block_15_add")([adding14, layer])

layer = tensorflow.keras.layers.Conv2D(336, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_16_expand")(adding15)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_16_expand_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_16_expand_relu")(layer)
layer = tensorflow.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides = (1, 1), padding = "same", data_format = "channels_last", activation="linear", groups= 1, depthwise_initializer= initializer, use_bias= False, name="block_16_depthwise")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_16_depthwise_BN")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="block_16_depthwise_relu")(layer)
layer = tensorflow.keras.layers.Conv2D(112, kernel_size=(1,1), strides=(1,1), padding = "same", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="block_16_project")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="block_16_project_BN")(layer)

layer = tensorflow.keras.layers.Conv2D(1280, kernel_size=(1,1), strides=(1,1), padding = "valid", data_format= 'channels_last', activation="linear", groups= 1, use_bias= False, kernel_initializer=initializer, name="Conv_1")(layer)
layer = tensorflow.keras.layers.BatchNormalization(axis=3, name="Conv_1_bn")(layer)
layer = tensorflow.keras.layers.ReLU(max_value=6.0, name="out_relu")(layer)

layer = tensorflow.keras.layers.GlobalAveragePooling2D(data_format= 'channels_last', name="global_average_pooling2d_GlobalAveragePooling2D1")(layer)

final_initializer = tensorflow.keras.initializers.VarianceScaling(scale=1, mode='fan_in', distribution='truncated_normal')

layer = tensorflow.keras.layers.Dense(units= 100, activation ="relu", kernel_initializer=final_initializer, name="dense_Dense1")(layer)
output_layer = tensorflow.keras.layers.Dense(units= 2, activation ="softmax", kernel_initializer=final_initializer, name="output_layer")(layer)


model = tensorflow.keras.models.Model(inputs=[input], outputs=output_layer)

model.summary()