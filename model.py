
import tensorflow as tf

# Import all remaining packages
import numpy as np
import os
import time


# Find all unique characters in the joined string
vocab = ['\n',
         ' ',
         '!',
         '"',
         '#',
         "'",
         '(',
         ')',
         ',',
         '-',
         '.',
         '/',
         '0',
         '1',
         '2',
         '3',
         '4',
         '5',
         '6',
         '7',
         '8',
         '9',
         ':',
         '<',
         '=',
         '>',
         'A',
         'B',
         'C',
         'D',
         'E',
         'F',
         'G',
         'H',
         'I',
         'J',
         'K',
         'L',
         'M',
         'N',
         'O',
         'P',
         'Q',
         'R',
         'S',
         'T',
         'U',
         'V',
         'W',
         'X',
         'Y',
         'Z',
         '[',
         ']',
         '^',
         '_',
         'a',
         'b',
         'c',
         'd',
         'e',
         'f',
         'g',
         'h',
         'i',
         'j',
         'k',
         'l',
         'm',
         'n',
         'o',
         'p',
         'q',
         'r',
         's',
         't',
         'u',
         'v',
         'w',
         'x',
         'y',
         'z',
         '|']
print("There are", len(vocab), "unique characters in the dataset")


char2idx = {u: i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)


def vectorize_string(string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output


def get_batch(vectorized_songs, seq_length, batch_size):
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_length, batch_size)

    '''TODO: construct a list of input sequences for the training batch'''
    input_batch = [vectorized_songs[i: i+seq_length] for i in idx]
    # input_batch = # TODO
    '''TODO: construct a list of output sequences for the training batch'''
    output_batch = [vectorized_songs[i+1: i+seq_length+1] for i in idx]
    # output_batch = # TODO

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch


# x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)


def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # Layer 1: Embedding layer to transform indices into dense vectors
        #   of a fixed embedding size
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

        # Layer 2: LSTM with `rnn_units` number of units.
        # TODO: Call the LSTM function defined above to add this layer.
        LSTM(rnn_units),
        # LSTM('''TODO'''),

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
        #   into the vocabulary size.
        # TODO: Add the Dense layer.
        tf.keras.layers.Dense(vocab_size)
        # '''TODO: DENSE LAYER HERE'''
    ])

    return model


#model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)

# model.summary()


# x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
#pred = model(x)


def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)
    # loss = tf.keras.losses.sparse_categorical_crossentropy('''TODO''', '''TODO''', from_logits=True) # TODO
    return loss


# example_batch_loss = compute_loss(y, pred)

# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters:
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location:
checkpoint_dir = './'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")


#model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate)


'''
@tf.function
def train_step(x, y):
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:

        #TODO: feed the current input into the model and generate predictions
        y_hat = model(x)  # TODO

        loss = compute_loss(y, y_hat)  # TODO


    # Now, compute the gradients

    grads = tape.gradient(loss, model.trainable_variables)  # TODO

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
'''

##################
# Begin training!#
##################


history = []
# plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
# if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists


'''
for iter in tqdm(range(num_training_iterations)):

    # Grab a batch and propagate it through the network
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    loss = train_step(x_batch, y_batch)

    # Update the progress bar
    history.append(loss.numpy().mean())

    # Update the model with the changed weights!
    if iter % 100 == 0:
        # model.save_weights(checkpoint_prefix)

        # Save the trained model and the weights

model.save("trained_model.h5")
'''

#model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)  #

# model.load_weights(tf.train.latest_checkpoint(checkpoint_prefix))
model = tf.keras.models.load_model("trained_model.h5")
# model.build(tf.TensorShape([1, None]))


def generate_text(start_string="X", generation_length=200):
    # Evaluation step (generating ABC text using the learned RNN model)
    '''TODO: convert the start string to numbers(vectorize)'''
    input_eval = [char2idx[s] for s in start_string]  # TODO
    # input_eval = ['''TODO''']
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()

    for i in range(generation_length):
        '''TODO: evaluate the inputs and generate the next character predictions'''
        predictions = model(input_eval)
        # predictions = model('''TODO''')

        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        '''TODO: use a multinomial distribution to sample'''
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        # predicted_id = tf.random.categorical('''TODO''', num_samples=1)[-1,0].numpy()

        # Pass the prediction along with the previous hidden state
        #   as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        '''TODO: add the predicted character to the generated text!'''
        # Hint: consider what format the prediction is in vs. the output
        text_generated.append(idx2char[predicted_id])  # TODO
        # text_generated.append('''TODO''')

    return (start_string + ''.join(text_generated))


generated_text = generate_text(start_string="X", generation_length=1000)
