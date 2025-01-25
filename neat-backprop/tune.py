from jax.nn import log_softmax
import jax.numpy as jnp
from fineNeat.sneat_jax.ann import act
from jax import value_and_grad
from jax.nn import softmax
from datagen import DataGenerator 

def one_hot_encode(batch_output):
    encoder = jnp.zeros((batch_output.shape[0], 2))
    encoder = encoder.at[jnp.arange(batch_output.shape[0]), batch_output.astype(int).reshape(-1)].set(1)
    return encoder

# simple loss function :: easier for gradient descent to occur for this simple loss functional
def loss_fn(wMat, aVec, nInput, nOutput, inputs, targets):
    logits = act(wMat, aVec, nInput, nOutput, inputs)
    one_hot_targets = one_hot_encode(targets)
    loss = jnp.mean((logits - one_hot_targets) ** 2)
    return loss


# Jax backprop step function
def step(wMat, aVec, nInput, nOutput, batch_input, batch_output, learning_rate=0.01):
    loss_value, grads = value_and_grad(loss_fn)(wMat, aVec, nInput, nOutput, batch_input, batch_output)
    wMat_updated = wMat - learning_rate * grads
    return wMat_updated, loss_value


def train_params(wMat, aVec, nInput, nOutput, train_data, generator: DataGenerator, learning_rate: float = 0.01, n_epochs: int = 400, interval: int = 10):
    for i in range(n_epochs): 
        batch = generator.generate_batch(train_data)
        batch_input, batch_output = batch[:, :2], batch[:, 2:]
        wMat, loss_value = step(wMat, aVec, nInput, nOutput, batch_input, batch_output, learning_rate=learning_rate)
        if i % interval == 0:
            print(f"Epoch {i + 1}, Loss: {loss_value}")
    return wMat, loss_value


# Deprecated loss_fn with cross entropy -- does not work well in this case 

# def cross_entropy_loss(logits, labels):
#     """
#     Args:
#         logits: raw output from act function [batch_size, n_classes]
#         labels: true labels [batch_size] or one-hot [batch_size, n_classes]
#     """
#     # Apply log_softmax to get log-probabilities
#     log_probs = log_softmax(logits)
    
#     # If labels are not one-hot, convert to one-hot
#     if labels.ndim == 1:
#         labels = jnp.eye(logits.shape[1])[labels]
    
#     # Compute cross entropy loss
#     loss = -jnp.sum(labels * log_probs) / labels.shape[0]
#     return loss

# def loss_fn(weights, aVec, nInput, nOutput, inputs, targets):
#     logits = act(weights, aVec, nInput, nOutput, inputs)
#     return cross_entropy_loss(logits, targets)