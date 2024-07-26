from src.util.board import *
from abc import ABC, abstractmethod
import tf_keras as keras
import tensorflow as tf


class Weapon(ABC):
    pass


class Core(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(8, 2, strides=2, padding='same')
        self.conv2 = keras.layers.Conv2D(32, 2, strides=2, padding='same')
        self.conv3 = keras.layers.Conv2D(64, 4, strides=2, padding='same')
        self.dense1 = keras.layers.Dense(units=128)
        self.dense2 = keras.layers.Dense(units=32)
        self.dense3 = keras.layers.Dense(units=1)
        self.flatten = keras.layers.Flatten()
    
    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.flatten(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    def children(self):
        return [self.conv1, self.conv2, self.conv3,
                self.dense1, self.dense2, self.dense3]
    

class Core_v2(keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv1 = keras.layers.Conv2D(8, 2, strides=2, padding='same')
        self.conv2 = keras.layers.Conv2D(32, 2, strides=2, padding='same')
        self.conv3 = keras.layers.Conv2D(64, 4, strides=2, padding='same')
        self.dense1 = keras.layers.Dense(units=128)
        self.dense2 = keras.layers.Dense(units=32)
        self.dense3 = keras.layers.Dense(units=19 * 19, activation='softmax')
        self.flatten = keras.layers.Flatten()
    
    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.flatten(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x 


class Katana(Weapon):
    def __init__(self):
        super().__init__()
        self.core = Core()
    
    def __mul__(self, other):
        
        def list_core(c1: Core, c2: Core):
            zipped = list(zip(c1.children(), c2.children()))
            return zipped

        def mutate_weight(c1: Core, c2: Core):
            zipped = list_core(c1, c2)
            zipped_weight = [(u.get_weights()[0], v.get_weights()[0]) for u, v in zipped]
            
            def mutate_matrix(m1: np.ndarray, m2: np.ndarray):
                shape = m1.shape 
                m = tf.where(tf.random.uniform(shape=shape) > 0.5, m1, m2)
                mx = tf.where(tf.random.uniform(shape) > 0.01, m, tf.random.uniform(shape))
                return mx

            final_scale = [mutate_matrix(u, v) for u, v in zipped_weight]

            return final_scale
        
        def forming(c1: Core, c2: Core):
            new_core = Core()
            new_weight = mutate_weight(c1, c2)
            for para, mutated in zip(new_core.children(), new_weight):
                para: keras.layers.Layer
                para.add_weight(shape=mutated.shape)
                para.set_weights([mutated])
            return new_core 
        
        core = forming(self.core, other.core)
        return core 
    
    def mate(self, c):
        return self * c 

    def __call__(self, inp):
        return self.core(inp)




        
