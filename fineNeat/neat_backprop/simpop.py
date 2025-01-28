from .tune import train_ind, get_reward
import jax

class SimplePop: 
    
    def __init__(self, ind, pop_size, hyp, train_data, test_data, generator): 
        self.ind = ind
        self.pop_size = pop_size
        self.reward = None
        self.hyp = hyp
        self.gen = 1 
        self.train_data = train_data
        self.test_data = test_data
        self.generator = generator
        self.population = self.init_pop()
        self.best_ind = None 
        self.best_fitness = 0.0

    def init_pop(self): 
        return [self.ind.mutate(self.hyp)[0] for i in range(self.pop_size)]

    def tell_pop(self, reward, k=3): 
        for i in range(len(reward)): 
            self.population[i].fitness = reward[i]
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        top_k = sorted_pop[:k]
        print(f"Top {k} network topology fitness at generation {self.gen}: ", [ind.fitness for ind in top_k])
        self.update_best_ind(top_k)
        return top_k
    
    def update_best_ind(self, top_k): 
        if top_k[0].fitness > self.best_fitness: 
            self.best_ind = top_k[0]
            self.best_fitness = self.best_ind.fitness
    
    def randomize_weights(self, pop): 
        randomized_pop = []
        for i, ind in enumerate(pop): 
            key = jax.random.PRNGKey(i)
            ind.conn = ind.conn.at[3, :].set(jax.random.normal(key, shape=(ind.conn[3].shape)) * 1.0)
            randomized_pop.append(ind)
        reward = get_reward(randomized_pop, self.test_data, nInput=2, nOutput=2)
        for i in range(len(reward)): 
            randomized_pop[i].fitness = reward[i]
        return randomized_pop
    
    def train_n_pick(self, top_k, k, n_epochs=800): 
        trained_pop = []
        for i in range(len(top_k)): 
            trained_ind, _ = train_ind(top_k[i], self.train_data, self.generator, learning_rate=0.01, n_epochs=n_epochs, interval=50, nInput=2, nOutput=2)
            trained_pop.append(trained_ind)
        backprop_reward = get_reward(trained_pop, self.test_data, nInput=2, nOutput=2)
        print("Backprop reward: ", backprop_reward)
        for i in range(len(backprop_reward)): 
            trained_pop[i].fitness = backprop_reward[i]
        sorted_trained_pop = sorted(trained_pop, key=lambda x: x.fitness, reverse=True)
        self.update_best_ind(sorted_trained_pop)
        print(f"Top {k} backprop fitness at generation {self.gen}: ", [ind.fitness for ind in sorted_trained_pop[:k]])
        return sorted_trained_pop[:k]

    # populate next generation
    def update_pop(self, top_k, power=1, refresh_pop: bool = False, randomize_weights: bool = False):
        if refresh_pop: 
            self.population = []
        else:
            self.population = top_k 
        
        # This is strangely slow ... why? 
        from tqdm import tqdm 
        print(f"Initializing generation {self.gen + 1} from top k individuals")
        for i in tqdm(range(self.pop_size - len(self.population))):
            child = top_k[i % len(top_k)] 
            try: 
                child, _ = child.mutate(self.hyp, seed=i)
            except: 
                print(":: Mutation failed")
                continue 
                
            self.population.append(child)
            
        if randomize_weights: 
            self.population = self.randomize_weights(self.population)
            
        self.gen += 1