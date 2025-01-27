from fineNeat import loadHyp, updateHyp, load_cls_task
from fineNeat.sneat_jax.ind import Ind 
from neat_backprop.datagen import DataGenerator
from neat_backprop.tune import train_ind, get_reward
from typing import Optional, List, Tuple
import os 
from tqdm import tqdm 
from jax import numpy as jnp 
import numpy as np 
from fineNeat import viewInd, fig2img
import matplotlib.pyplot as plt 
from neat_backprop.viz import plot_decision_boundary
import argparse
import random


def main(args):
    
    choice_id = args.taskid
    generator = DataGenerator(train_size=2000, batch_size=1000)
    train_data, test_data = generator.generate_random_dataset(choice=choice_id)  # 0 for circle dataset

    hyp_default = '../fineNeat/fineNeat/p/cls.json'
    hyp_adjust = '../fineNeat/fineNeat/p/cls_neat.json'

    hyp = loadHyp(pFileName=hyp_default, load_task=load_cls_task)
    updateHyp(hyp,load_cls_task,hyp_adjust)

    best_ind = None 
    learning_rate = 0.01
    n_epochs = 800
    nInput, nOutput = 2, 2

    def backprop_top_inds(top_inds: List[Ind], learning_rate: float, n_epochs: int, interval: int=50, nInput: int=2, nOutput: int=2):
        best_reward = 0.0 
        best_ind = None 
        
        for ind in top_inds: 
            new_ind = train_ind(ind, train_data, generator, learning_rate=learning_rate, n_epochs=n_epochs, interval=interval, nInput=nInput, nOutput=nOutput)[0]
            reward_value = get_reward([new_ind], test_data, nInput, nOutput)
            new_ind.fitness = reward_value[0]
            if new_ind.fitness > best_reward: 
                best_reward = new_ind.fitness
                best_ind = new_ind 

        return best_ind 

    # Create directories
    logdir = args.logdir
    visdir = os.path.join(logdir, "vis")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(visdir, exist_ok=True)

    # Initialize Population from Base Shape
    init_shape = [(2, 5), (5, 2)]
    population = [Ind.from_shapes(init_shape) for _ in range(args.population_size)]
    for ind in population: 
        ind.express()

    # Initialize Winning Streak
    winning_streak = [0] * args.population_size

    for tournament in tqdm(range(1, args.total_tournaments+1)):
        left_idx, right_idx = random.sample(range(args.population_size), 2)
        left_ind, right_ind = population[left_idx], population[right_idx]

        # Evaluate the fitness of the two individuals   
        rewards = get_reward([left_ind, right_ind], test_data, nInput, nOutput)
        left_ind.fitness = rewards[0]
        right_ind.fitness = rewards[1]
        
        # Population update 
        if left_ind.fitness == right_ind.fitness: 
            population[left_idx] = left_ind.mutate(hyp)[0]
        elif left_ind.fitness < right_ind.fitness: 
            population[left_idx] = right_ind.mutate(hyp)[0]
            winning_streak[left_idx] = winning_streak[right_idx]
            winning_streak[right_idx] += 1 
        else: 
            population[right_idx] = left_ind.mutate(hyp)[0]
            winning_streak[right_idx] = winning_streak[left_idx]
            winning_streak[left_idx] += 1 
            
            
        if tournament % args.save_freq == 0:
            model_filename = os.path.join(logdir, f"sneat_cls_neat_{tournament:08d}.json")
            with open(model_filename, 'wt') as out:
                record_holder = np.argmax(winning_streak)
                population[record_holder].save(model_filename)

        if (tournament) % args.save_freq == 0:
            record_holder = np.argmax(winning_streak)
            fig, _ = viewInd(population[record_holder])
            plt.close(fig)
            img = fig2img(fig)
            img.save(os.path.join(visdir, f"sneat_cls_neat_{tournament:08d}.png"))
            
            # Plot decision boundary
            fig, ax = plot_decision_boundary(population[record_holder].wMat, population[record_holder].aVec, nInput, nOutput, test_data)
            plt.close(fig)
            img = fig2img(fig)
            img.save(os.path.join(visdir, f"sneat_cls_neat_decision_boundary_{tournament:08d}.png"))
            
            record = winning_streak[record_holder]
            print(f"tournament: {tournament}, best_winning_streak: {record}")
            
    # BackProp on record holder individuals | 10 tries 
    print(" :: Start BackProp on top individual for 10 times ...")
    best_inds = backprop_top_inds([population[record_holder]] * 10, learning_rate, n_epochs, interval=50, nInput=2, nOutput=2)
    best_ind = max(best_inds, key=lambda x: x.fitness)
    
    # Save checkpoint and best individual image 
    model_filename = os.path.join(logdir, f"sneat_cls_backprop_best.json")
    best_ind.save(model_filename)
    fig, _ = viewInd(best_ind)
    plt.close(fig)
    img = fig2img(fig)
    img.save(os.path.join(visdir, f"sneat_cls_backprop_best.png"))
    
    # Plot decision boundary
    fig, ax = plot_decision_boundary(best_ind.wMat, best_ind.aVec, nInput, nOutput, test_data)
    img = fig2img(fig)
    img.save(os.path.join(visdir, f"sneat_cls_backprop_best_decision_boundary.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SNEAT cls tuning script')
    parser.add_argument('--seed', type=int, default=612, help='Random seed')
    parser.add_argument('--population-size', type=int, default=128, help='Population size')
    parser.add_argument('--total-tournaments', type=int, default=10000, help='Total number of tournaments')
    parser.add_argument('--save-freq', type=int, default=1000, help='Save frequency')
    parser.add_argument('--hyp-default', type=str, default='fineNeat/p/default_sneat.json', help='Default hyperparameters file')
    parser.add_argument('--hyp-adjust', type=str, default='fineNeat/p/volley_sparse.json', help='Adjustment hyperparameters file')
    parser.add_argument('--logdir', type=str, default='../runs/sneat_cls_tune', help='Log directory')
    parser.add_argument('--taskid', type=int, default=0, help='Task ID')
    args = parser.parse_args()
    main(args)