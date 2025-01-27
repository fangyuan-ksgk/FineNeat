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
generator = DataGenerator(train_size=2000, batch_size=1000)
train_data, test_data = generator.generate_random_dataset(choice=choice_id)  # 0 for circle dataset


hyp_default = '../fineNeat/fineNeat/p/cls.json'
hyp_adjust = '../fineNeat/fineNeat/p/cls_neat.json'
fileName = "cls"

hyp = loadHyp(pFileName=hyp_default, load_task=load_cls_task)
updateHyp(hyp,load_cls_task,hyp_adjust)

# save best ind 
log_dir = "../runs/simple_cls"
vis_dir = os.path.join(log_dir, "vis")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)


nInput = 2
nOutput = 2
ind = Ind.from_shapes([(nInput,nOutput)], act_id=9)
ind.express()

pop_size = 256

from neat_backprop.simpop import SimplePop
# Initialize Population 

pop = SimplePop(ind, pop_size, hyp, train_data, test_data, generator)

for iter in range(120): 
    reward = get_reward(pop.population, test_data, nInput=2, nOutput=2)
    top_k = pop.tell_pop(reward, k=9)
    top_k = pop.train_n_pick(top_k, k=3)
    pop.update_pop(top_k, power = (iter+1)//2)
    
    # Save best individual with iteration number
    best_ind = top_k[0]
    best_ind.save(f"{log_dir}/best_ind_in_pop_{fileName}_generation_{iter}.json")
    fig, ax = plot_decision_boundary(best_ind.wMat, best_ind.aVec, nInput=2, nOutput=2, data=test_data)
    plt.close(fig)
    img = fig2img(fig)
    # Save visualization with iteration number
    img.save(os.path.join(vis_dir, f"best_ind_decision_boundary_generation_{iter}.png"))
    fig, ax = viewInd(best_ind)
    plt.close(fig)
    img = fig2img(fig)
    img.save(os.path.join(vis_dir, f"best_ind_view_generation_{iter}.png"))