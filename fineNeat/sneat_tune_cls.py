from fineNeat import loadHyp, updateHyp, load_cls_task
from neat_backprop.simpop import SimplePop
from neat_backprop.datagen import DataGenerator
from neat_backprop.tune import get_reward
from fineNeat.sneat_jax.ind import Ind 
from neat_backprop.viz import plot_decision_boundary
import matplotlib.pyplot as plt
from fineNeat import fig2img, viewInd 
import argparse
import os 


def main(args): 
    
    choice_id = args.cls_id
    generator = DataGenerator(train_size=args.train_size, test_size=args.test_size, batch_size=args.batch_size)
    train_data, test_data = generator.generate_random_dataset(choice=choice_id)  # 0 for circle dataset
    hyp_default = args.hyp_default_config 
    hyp_adjust = args.hyp_adjust_config 
    fileName = "cls"

    hyp = loadHyp(pFileName=hyp_default, load_task=load_cls_task)
    updateHyp(hyp,load_cls_task,hyp_adjust)

    # save best ind 
    log_dir = args.log_dir
    vis_dir = os.path.join(log_dir, "vis")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    nInput = 2
    nOutput = 2
    ind = Ind.from_shapes([(nInput,nOutput)], act_id=9)
    ind.express()

    pop_size = args.pop_size

    # Initialize Population 
    pop = SimplePop(ind, pop_size, hyp, train_data, test_data, generator)
    
    for iter in range(args.n_iter): 
        reward = get_reward(pop.population, test_data, nInput=2, nOutput=2)
        top_k = pop.tell_pop(reward, k=args.topology_pick)
        top_k = pop.train_n_pick(top_k, k=args.backprop_pick, n_epochs=args.backprop_epochs)

        # Simplified update logic to match train_simple_cls.py
        perturb_update = True if (iter>0 and iter % args.refresh_pop_interval==0) else False
        pop.update_pop(top_k, power=1, perturb_update=perturb_update)
        
        # Save visualization code remains unchanged
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
        
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_id', type=int, default=0, help='Choice of Classification Task')
    parser.add_argument('--hyp_default_config', type=str, default='./fineNeat/p/cls.json', help='Default configuration file for NEAT')
    parser.add_argument('--hyp_adjust_config', type=str, default='./fineNeat/p/cls_neat.json', help='Adjusted configuration file for NEAT')
    parser.add_argument('--log_dir', type=str, default='../runs/sneat_tune_cls', help='Directory to save logs and visualizations')
    parser.add_argument('--pop_size', type=int, default=64, help='Population size')
    parser.add_argument('--backprop_epochs', type=int, default=800, help='Number of epochs for backpropagation')
    parser.add_argument('--n_iter', type=int, default=120, help='Number of iterations')
    parser.add_argument('--topology_pick', type=int, default=9, help='Number of top individuals to pick for topology update')
    parser.add_argument('--backprop_pick', type=int, default=3, help='Number of top individuals to pick for backpropagation')
    parser.add_argument('--refresh_pop_interval', type=int, default=3, help='Interval for refreshing population')
    parser.add_argument('--train_size', type=int, default=2000, help='Training data size')
    parser.add_argument('--test_size', type=int, default=500, help='Testing data size')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    args = parser.parse_args()
    main(args)