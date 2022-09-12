############################
# Self-explanatory imports #
############################
import tensorflow as tf
import numpy as np

####################
# Limits GPU usage #
####################
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*4)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#################################
# Generate noisy parabolic data #
#################################

# Generates 201 evenly spaced values between -2 and 2
x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

# Just some function we're plotting, in this case it's a parabola
def f(x):
  y = x**2 + 2*x - 5
  return y

# the random.normal code produces 201 (given by shape) points as a vector following a
# normal distribution with mean 0 and standard deviation 1 (default values)
y = f(x) + tf.random.normal(shape=[201])


#########################################
# Create model and biweekly report data #
#########################################

def train_models(write_lines, shuffle = True, n = 1):
    for i in range(n):
        model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.stack([x**2, x], axis=1)),
            tf.keras.layers.Dense(1, kernel_initializer=tf.random.normal)])

        model.compile(
            loss=tf.keras.losses.MSE,
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))

        # Run the training
        history = model.fit(x, y,
                                epochs=100,
                                batch_size=32,
                                shuffle=shuffle,
                                verbose=0)
        
        print(f'Final mean squared error for shuffle {shuffle} model {i}: {history.history["loss"][-1]:0.3f}')
        write_lines.append(f'{history.history["loss"][-1]:0.6f}\n')


file = open("MSE_True.txt", "w")
file.seek(0)
write_lines = []
train_models(write_lines, shuffle = True, n = 100)
file.writelines(write_lines)
file.close()

file = open("MSE_False.txt", "w")
file.seek(0)
write_lines = []
train_models(write_lines, shuffle = False, n = 100)
file.writelines(write_lines)
file.close()
