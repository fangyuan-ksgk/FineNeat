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


def train_species(self, species): 
    species = self.species 
    spec_individuals = [spec.seed for spec in species]
    from neat_backprop.tune import train_ind 
    trained_individuals = []
    for ind in spec_individuals: 
        trained_ind, _ = train_ind(ind, train_data, generator, learning_rate=0.01, n_epochs=800, interval=50, nInput=2, nOutput=2)
        trained_individuals.append(trained_ind)
    for i in range(len(species)): 
        self.species[i].seed = trained_individuals[i]
        self.species[i].bestInd = trained_individuals[i]
        self.species[i].members.append(trained_individuals[i])
    self.pop += trained_individuals 
    

def print_best_individual(neat, iter): 
    best_ind = sorted(neat.pop, key=lambda x: x.fitness, reverse=True)[0]
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
for iter in range(97): 
    pop = neat.ask()        
    reward = get_reward(pop, test_data, nInput=2, nOutput=2)
    print("Best reward: ", max(reward))
    neat.tell(reward)
    
    if iter > 0 and iter % 3 == 0: 
        train_species(neat, neat.species)
        print_best_individual(neat, iter)


