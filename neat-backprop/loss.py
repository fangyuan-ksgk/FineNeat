from jax.nn import log_softmax
import jax.numpy as jnp

def cross_entropy_loss(logits, labels):
    """
    Args:
        logits: raw output from act function [batch_size, n_classes]
        labels: true labels [batch_size] or one-hot [batch_size, n_classes]
    """
    # Apply log_softmax to get log-probabilities
    log_probs = log_softmax(logits)
    
    # If labels are not one-hot, convert to one-hot
    if labels.ndim == 1:
        labels = jnp.eye(logits.shape[1])[labels]
    
    # Compute cross entropy loss
    loss = -jnp.sum(labels * log_probs) / labels.shape[0]
    return loss