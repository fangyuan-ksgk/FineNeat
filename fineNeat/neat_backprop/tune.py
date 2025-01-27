from jax.nn import log_softmax
import jax.numpy as jnp
from fineNeat.sneat_jax.ann import act, getMat, getNodeInfo
from jax import value_and_grad
from jax.nn import softmax
from .datagen import DataGenerator 
# from fineNeat import update_conn

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

# Connection Weight to Loss
def conn_weight_to_loss(conn_weight, wMat, aVec, nInput, nOutput,src_seqs, dest_seqs, batch_input, batch_output): 
    wMat_new = wMat.at[src_seqs, dest_seqs].set(conn_weight)
    loss_value = loss_fn(wMat_new, aVec, nInput, nOutput, batch_input, batch_output)
    return loss_value

def step_conn_weight(conn_weight, wMat, aVec, nInput, nOutput, src_seqs, dest_seqs,batch_input, batch_output, learning_rate=0.01): 
    loss_value, grads = value_and_grad(conn_weight_to_loss)(conn_weight, wMat, aVec, nInput, nOutput, src_seqs, dest_seqs, batch_input, batch_output)
    conn_weight_updated = conn_weight - learning_rate * grads
    return conn_weight_updated, loss_value

from tqdm import tqdm 

def train_ind(ind, train_data, generator: DataGenerator, learning_rate=0.01, n_epochs=400, interval=50, nInput=2, nOutput=2):
    wMat, node2order, node2seq, seq2node = getMat(jnp.array(ind.node), jnp.array(ind.conn))
    src_nodes, dest_nodes = ind.conn[1,:].astype(int), ind.conn[2,:].astype(int)
    src_seqs = [node2seq[src_node.item()] for src_node in src_nodes]
    dest_seqs = [node2seq[dest_node.item()] for dest_node in dest_nodes]
    aVec = jnp.array(ind.aVec)
    
    pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training")
    for i in pbar: 
        batch = generator.generate_batch(train_data)
        batch_input, batch_output = batch[:, :2], batch[:, 2:]
        conn_weight, loss_value = step_conn_weight(ind.conn[3,:], wMat, aVec, nInput, nOutput, src_seqs, dest_seqs, batch_input, batch_output, learning_rate=learning_rate)
        ind = update_ind(ind, conn_weight)
        if i % interval == 0:
            pbar.set_description(f"Training (Loss: {loss_value:.4f})")
            
    ind = update_ind(ind, conn_weight)
    return ind, loss_value
        

def update_ind(ind, conn_weight): 
    ind.conn = jnp.array(ind.conn).at[3,:].set(conn_weight)
    ind.express()
    return ind 


def get_reward(pop, test_data, nInput=2, nOutput=2): 
    reward = []
    for ind in pop: 
        loss = loss_fn(ind.wMat, ind.aVec, nInput, nOutput, test_data[:,:2], test_data[:,2])
        reward_value = 1 - loss.item()
        reward.append(reward_value)
    return reward

def backprop_per_species(neat, train_data, test_data, generator: DataGenerator, n_top_seed=8, learning_rate=0.01, n_epochs=400, nInput=2, nOutput=2):
    """ 
    For each species, do backprop on the best individual therein, and output the best individual
    """

    top_species = sorted(neat.species, key=lambda x: x.seed.fitness, reverse=True)[:n_top_seed]

    top_individuals = []
    for s in top_species: 
        raise ValueError("Not implemented")
        seed = s.bestInd
        wMat = jnp.copy(seed.wMat)
        aVec = jnp.array(seed.aVec)
        wMat, _ = train_params(wMat, aVec, nInput, nOutput, train_data, generator, learning_rate=learning_rate, n_epochs=n_epochs, interval=50)

        reward_value = get_reward([seed], test_data, nInput, nOutput)
        seed.fitness = reward_value[0]
        seed = update_conn(seed, wMat)
        top_individuals.append(seed)

    best_ind = sorted(top_individuals, key=lambda x: x.fitness, reverse=True)[0]
    
    return best_ind 