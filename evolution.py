from player import Player
import numpy as np
import json
import copy


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.generation_number = 0
    
               
       
    def roulette_wheel(self, players, parent_numbers):
        
        total_fitness = sum([player.fitness for player in players])
        probabilities = [player.fitness/total_fitness for player in players]            
        
        for i in range(1, len(players)):
            probabilities[i] += probabilities[i - 1]
              
        results = []
        for random_number in np.random.uniform(low=0, high=1, size=parent_numbers):
            for i, probability in enumerate(probabilities):
                if random_number <= probability:
                    results.append(players[i])
                    break

        return results

       
    def q_tournament(self, players, q):
        q_sel = np.random.choice(players, q)
        return max(q_sel, key=lambda player: player.fitness)  

    def sus(self, players, num_players):
        step = 1 - 1 / num_players
        steps = np.linspace(0, step, num_players)
        random_number = np.random.uniform(0, 1 / num_players, 1)
        steps += random_number
        
        total_fitness = sum([player.fitness for player in players])
        probabilities = [player.fitness/total_fitness for player in players]            
        
        for i in range(1, len(players)):
            probabilities[i] += probabilities[i - 1]         

        result = []
        for interval in steps:
            for i, probability in enumerate(probabilities):
                if interval < probability:
                    result.append(players[i])
                    break

        return result

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # top-k
        players = sorted(players, key=lambda player: player.fitness, reverse=True)
            
        # roulette wheel
#        players = self.roulette_wheel(players, num_players)
        
        # SUS
#        players = self.sus(players, num_players)        
        
        # Q tournament
#        result = []
#        for _ in range(num_players):
#            result.append(self.q_tournament(players, q=3))
#        players = result


        fitness_list = [player.fitness for player in players]
        max_fitness = float(np.max(fitness_list))
        mean_fitness = float(np.mean(fitness_list))
        min_fitness = float(np.min(fitness_list))
        self.save_fitness_result(min_fitness, max_fitness, mean_fitness)    
        
        
        return players[: num_players]
    

    def save_fitness_result(self, min_fitness, max_fitness, mean_fitness):
        if self.generation_number == 0:
            fitness_results = {
                'min': [min_fitness],
                'max': [max_fitness],
                'mean': [mean_fitness]
            }
            file1 = open('myfile.txt', 'w', encoding='utf_8')
            file1.write(json.dumps(fitness_results))
            file1.close()
        else:
            file = open('myfile.txt', encoding='utf8')
            read = file.read()
            fitness_results = json.loads(read)         

            fitness_results['min'].append(min_fitness)
            fitness_results['max'].append(max_fitness)
            fitness_results['mean'].append(mean_fitness)
            
            file1 = open('myfile.txt', 'w', encoding='utf_8')
            file1.write(json.dumps(fitness_results))
            file1.close()


        self.generation_number += 1
        


    
    def generate(self, parent1, parent2):
        child1 = self.clone_player(parent1)
        child2 = self.clone_player(parent2)
        
        
        for layer in range(0,len(parent1.nn.W)):
            self.crossover(child1.nn.W[layer], child2.nn.W[layer],
                           parent1.nn.W[layer], parent2.nn.W[layer])
                    
            
        for bias in range(0,len(parent1.nn.b)):
            self.crossover(child1.nn.b[bias], child2.nn.b[bias],
                           parent1.nn.b[bias], parent2.nn.b[bias])
        

        self.mutate(child1)
        self.mutate(child2)

        return child1, child2
    

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """        
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for random_player in range(num_players)]
        
        else:            
            parents = []

        
            # SUS
#            parents = self.sus(prev_players, num_players)              

            # roulette wheel
            parents = self.roulette_wheel(prev_players, num_players)

            # Q tournament
#            for _ in range(num_players):
#                parents.append(self.q_tournament(prev_players, q=3))
                    
                    
            new_players = []

            for i in range(0, len(parents), 2):
                child1, child2 = self.generate(parents[i], parents[i + 1])
                new_players.append(child1)
                new_players.append(child2)

            return new_players             
            

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
    
   

    def crossover(self, child1_array, child2_array, parent1_array, parent2_array):
        n = child1_array.shape[0]
        
        random_number = np.random.uniform(0, 1, 1)
        
        if random_number <= 0.45:
            child1_array[0:n // 3] = parent1_array[0:n // 3]
            child1_array[n // 3:2*n//3] = parent2_array[n // 3:2*n//3]
            child1_array[2*n//3:n] = parent1_array[2*n//3:n]
        
            child2_array[0:n // 3] = parent2_array[0:n // 3]
            child2_array[n // 3:2*n//3] = parent1_array[n // 3:2*n//3]
            child2_array[2*n//3:n] = parent2_array[2*n//3:n]
            
        elif random_number > 0.45 and random_number <= 0.90:
            child2_array[0:n // 3] = parent1_array[0:n // 3]
            child2_array[n // 3:2*n//3] = parent2_array[n // 3:2*n//3]
            child2_array[2*n//3:n] = parent1_array[2*n//3:n]
        
            child1_array[0:n // 3] = parent2_array[0:n // 3]
            child1_array[n // 3:2*n//3] = parent1_array[n // 3:2*n//3]
            child1_array[2*n//3:n] = parent2_array[2*n//3:n]
            
        else:
            child1_array = parent1_array
            child2_array = parent2_array
            
     
    

    def mutate(self, child):        
        random_number = np.random.uniform(0, 1, 1)
        if random_number < 0.1:   
            for i in range(0,len(child.nn.W)):
                child.nn.W[i] += np.random.normal(size=child.nn.W[i].shape)
            for j in range(0,len(child.nn.b)):
                child.nn.b[j] += np.random.normal(size=child.nn.b[j].shape)
                
      

       
