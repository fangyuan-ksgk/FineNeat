from fineNeat import Neat 
from fineNeat import loadHyp, updateHyp, load_cls_task
from neat_backprop.datagen import DataGenerator
from neat_backprop.tune import get_reward
from neat_backprop.tune import get_reward, jnp, train_ind
from fineNeat.sneat_jax.ind import Ind 
import os
from neat_backprop.viz import plot_decision_boundary
import matplotlib.pyplot as plt
from fineNeat import fig2img, viewInd 
import os 


choice_id = 0
generator = DataGenerator(train_size=20000, batch_size=10000)
train_data, test_data = generator.generate_random_dataset(choice=choice_id)  # 0 for circle dataset


hyp_default = '../fineNeat/fineNeat/p/cls.json'
hyp_adjust = '../fineNeat/fineNeat/p/cls_neat.json'
fileName = "cls"

hyp = loadHyp(pFileName=hyp_default, load_task=load_cls_task)
updateHyp(hyp,load_cls_task,hyp_adjust)

# save best ind 
log_dir = "../runs/neat_simple_cls"
vis_dir = os.path.join(log_dir, "vis")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)


def train_best_species(neat, test_data, train_data, generator, iter, nInput=2, nOutput=2):

    best_species = sorted(neat.species, key=lambda x: x.bestFit, reverse=True)[:3]
    best_inds = [spec.bestInd for spec in best_species]

    # multiple training runs
    best_trained = [] 
    for best_ind in best_inds:
        min_loss = 1e10
        for _ in range(3): 
            _best_ind, loss_val = train_ind(best_ind, train_data, generator, learning_rate=0.005, n_epochs=2400, interval=50, nInput=2, nOutput=2)
            if loss_val < min_loss: 
                min_loss = loss_val
                best_trained_ind = _best_ind
                
        best_trained_ind.express()
        best_trained_ind.fitness = get_reward([best_trained_ind], test_data, nInput=2, nOutput=2)[0]
        best_trained.append(best_trained_ind)
        
    best_ind = sorted(best_trained, key=lambda x: x.fitness, reverse=True)[0]

    print(":: Best Individual Fitness: ", best_ind.fitness)

    # Save best individual with iteration number
    best_ind.save(f"{log_dir}/best_ind_{fileName}_iter_{iter}.json")

    fig, ax = plot_decision_boundary(best_ind.wMat, best_ind.aVec, nInput=2, nOutput=2, data=test_data)
    plt.close(fig)
    img = fig2img(fig)
    # Save visualization with iteration number
    img.save(os.path.join(vis_dir, f"best_ind_decision_boundary_iter_{iter}.png"))

    fig, ax = viewInd(best_ind)
    plt.close(fig)
    img = fig2img(fig)
    img.save(os.path.join(vis_dir, f"best_ind_view_iter_{iter}.png"))
    
    
# Iterate 2-stage neat-backprop 
# First Stage features NEAT topology search 
neat = Neat(hyp)

neat.initPop()
for iter in range(512): 
    pop = neat.ask()        
    reward = get_reward(pop, test_data, nInput=2, nOutput=2)
    print("Best reward: ", max(reward))
    neat.tell(reward)
    
    if iter % 16 == 0: 
        # pick best species and train them
        train_best_species(neat, test_data, train_data, generator, iter)


