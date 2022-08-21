import numpy as np
import pygame
from variables import global_variables
from nn import NeuralNetwork

def nn_input(screen_width, screen_height, obstacles, player_x, player_y):    
    norm_obs_x0 = 0
    norm_obs_y0 = 0
    
    norm_obs_x1 = 0
    norm_obs_y1 = 0
    
    norm_obs_x2 = 0
    norm_obs_y2 = 0
     
    norm_obs_x3 = 0
    norm_obs_y3 = 0  
    
    if len(obstacles) > 0 and obstacles[0]['y'] < player_y:
        norm_obs_x0 = float(obstacles[0]['x']) / screen_width
        norm_obs_y0 = float(obstacles[0]['y']+100) / screen_height
        
    if len(obstacles) > 1  and obstacles[1]['y'] < player_y:
        norm_obs_x1 = float(obstacles[1]['x']) / screen_width
        norm_obs_y1 = float(obstacles[1]['y']+100) / screen_height
        
    if len(obstacles) > 2  and obstacles[2]['y'] < player_y:
       norm_obs_x2 = float(obstacles[2]['x']) / screen_width
       norm_obs_y2 = float(obstacles[2]['y']+100) / screen_height

    if len(obstacles) > 3  and obstacles[3]['y'] < player_y:
       norm_obs_x3 = float(obstacles[3]['x']) / screen_width
       norm_obs_y3 = float(obstacles[3]['y']+100) / screen_height       
               
    x_feature = float(player_x-177) / (410-177)       
    
    x = np.array((
        [norm_obs_x0],
        [norm_obs_y0],
        [norm_obs_x1],
        [norm_obs_y1],
        [norm_obs_x2],
        [norm_obs_y2], 
        [norm_obs_x3],
        [norm_obs_y3],
        [x_feature]
        ))
    
 #   norm = np.linalg.norm(x)
 #   normal_x = x/norm
 
    return x 


 ####for batch normalization
#    gamma = 1
#    beta = 0
#    sample_mean = np.mean(x)
#    sample_var = np.var(x)
#    eps = 1e-8
#    x_normalized = (x - sample_mean)/np.sqrt(sample_var + eps)
#    out = gamma*x_normalized + beta 
    
#    return out


class Player(pygame.sprite.Sprite):
    def __init__(self, game_mode):
        super().__init__()

        # loading images
        player_walk1 = pygame.image.load('Graphics/Player/player_walk_1.png').convert_alpha()
        player_walk2 = pygame.image.load('Graphics/Player/player_walk_2.png').convert_alpha()

        # rotating -90 degree and scaling by factor of 0.5
        player_walk1 = pygame.transform.rotozoom(player_walk1, -90, 0.5)
        player_walk2 = pygame.transform.rotozoom(player_walk2, -90, 0.5)

        # flipping vertically
        player_walk1 = pygame.transform.flip(player_walk1, flip_x=False, flip_y=True)
        player_walk2 = pygame.transform.flip(player_walk2, flip_x=False, flip_y=True)

        self.player_walk = [player_walk1, player_walk2]

        self.player_index = 0

        self.image = self.player_walk[self.player_index]
        self.rect = self.image.get_rect(midleft=(177, 656))

        self.player_gravity = 'left'
        self.gravity = 10
        self.game_mode = game_mode

        if self.game_mode == "Neuroevolution":
            self.fitness = 0  # Initial fitness

            layer_sizes = [9, 20, 2]  
            #layer_sizes = [6, 15, 2]
            self.nn = NeuralNetwork(layer_sizes)

    def think(self, screen_width, screen_height, obstacles, player_x, player_y):                
        x = nn_input(screen_width, screen_height, obstacles, player_x, player_y)
        result = self.nn.forward(x)

        """
        Creates input vector of the neural network and determines the gravity according to neural network's output.

        :param screen_width: Game's screen width which is 604.
        :param screen_height: Game's screen height which is 800.
        :param obstacles: List of obstacles that are above the player. Each entry is a dictionary having 'x' and 'y' of
        the obstacle as the key. The list is sorted based on the obstacle's 'y' point on the screen. Hence, obstacles[0]
        is the nearest obstacle to our player. It is also worthwhile noting that 'y' range is in [-100, 656], such that
        -100 means it is off screen (Topmost point) and 656 means in parallel to our player's 'y' point.
        :param player_x: 'x' position of the player
        :param player_y: 'y' position of the player
        """

        # This is a test code that changes the gravity based on a random number. Remove it before your implementation.
        if result[0] >= 0.5:
            if max(result[0],result[1]) == result[0]:
                self.change_gravity('left')
            else:
                self.change_gravity('right')
        elif result[1] >= 0.5:
            if max(result[0],result[1]) == result[1]:
                self.change_gravity('right')
            else:
                self.change_gravity('left')  
                

    def change_gravity(self, new_gravity):
        """
        Changes the self.player_gravity based on the input parameter.
        :param new_gravity: Either "left" or "right"
        """
        new_gravity = new_gravity.lower()

        if new_gravity != self.player_gravity:
            self.player_gravity = new_gravity
            self.flip_player_horizontally()

    def player_input(self):
        """
        In manual mode: After pressing space from the keyboard toggles player's gravity.
        """
        if global_variables['events']:
            for pygame_event in global_variables['events']:
                if pygame_event.type == pygame.KEYDOWN:
                    if pygame_event.key == pygame.K_SPACE:
                        self.player_gravity = "left" if self.player_gravity == 'right' else 'right'
                        self.flip_player_horizontally()

    def apply_gravity(self):
        if self.player_gravity == 'left':
            self.rect.x -= self.gravity
            if self.rect.left <= 177:
                self.rect.left = 177
        else:
            self.rect.x += self.gravity
            if self.rect.right >= 430:
                self.rect.right = 430

    def animation_state(self):
        """
        Animates the player.
        After each execution, it increases player_index by 0.1. Therefore, after ten execution, it changes the
        player_index and player's frame correspondingly.
        """
        self.player_index += 0.1
        if self.player_index >= len(self.player_walk):
            self.player_index = 0

        self.image = self.player_walk[int(self.player_index)]

    def update(self):
        """
        Updates the player according to the game_mode. If it is "Manual", it listens to the keyboard. Otherwise the
        player changes its location based on `think` method.
        """
        if self.game_mode == "Manual":
            self.player_input()
        if self.game_mode == "Neuroevolution":
            obstacles = []
            for obstacle in global_variables['obstacle_groups']:
                if obstacle.rect.y <= 656:
                    obstacles.append({'x': obstacle.rect.x, 'y': obstacle.rect.y})

            self.think(global_variables['screen_width'],
                       global_variables['screen_height'],
                       obstacles, self.rect.x, self.rect.y)

        self.apply_gravity()
        self.animation_state()

    def flip_player_horizontally(self):
        """
        Flips horizontally to have a better graphic after each jump.
        """
        for i, player_surface in enumerate(self.player_walk):
            self.player_walk[i] = pygame.transform.flip(player_surface, flip_x=True, flip_y=False)
