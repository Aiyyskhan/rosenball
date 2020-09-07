import arcade as ar
import numpy as np

from neuralnetwork import *
from genetic_algorithm import *
from variables import *

W = 800
H = 600

SCALING_SPRITE = 0.5
SPEED = 5
JUMP_SPEED = 10
GRAVITY = 5
GAP_SIZE = 1
SPRITE_PIXEL_SIZE = 128
GRID_PIXEL_SIZE = (SPRITE_PIXEL_SIZE * SCALING_SPRITE)

START_STATE = 0
GAME_RUNNING = 1
GAME_OVER = 2

class Ball:
    def __init__(self, px, py, r):
        self.position_x = px
        self.position_y = py
        self.change_y = 0
        self.radius = r
        self.color = np.random.randint(0, 256, 3)
        self.distance_traveled = 0
        self.distance_to_wall = 0
        self.stop = False

    def draw(self):
        ar.draw_circle_filled(self.position_x, self.position_y, self.radius, self.color)

    def update(self):

        self.position_y += self.change_y

        self.distance_traveled += SPEED

class Wall:
    def __init__(self):
        self.wall_list = None
        self.wall_size = int((H // GRID_PIXEL_SIZE) + 1)
        self.prime_coord_list = [int(GRID_PIXEL_SIZE*(0.5 + i)) for i in range(self.wall_size)]
        self.wall_center_x = W + int(GRID_PIXEL_SIZE / 2)
        self.gap_center = None

    def setup(self):
        self.wall_list = ar.SpriteList()
        coord_list = self._gaping()
        for y in coord_list:
            brick = ar.Sprite('images/boxCrate_double.png', SCALING_SPRITE)
            brick.center_x = W + int(GRID_PIXEL_SIZE / 2)
            brick.center_y = y
            brick.change_x = -SPEED
            self.wall_list.append(brick)

    def draw(self):
        self.wall_list.draw()

    def update(self):
        self.wall_list.update()
        self.wall_center_x = self.wall_list[0].center_x
        
        if self.wall_center_x < (0 - int(GRID_PIXEL_SIZE / 2)):
            for brick in self.wall_list:
                brick.kill()
            self.setup()

    def _gaping(self):
        cut_coord_list = [i for i in self.prime_coord_list]
        rand_index = np.random.randint(0, self.wall_size - GAP_SIZE) #(1, self.wall_size + 1 - GAP_SIZE)
        self._gap_center_calculate(cut_coord_list[rand_index : rand_index + GAP_SIZE])
        del cut_coord_list[rand_index : rand_index + GAP_SIZE]
        return cut_coord_list

    def _gap_center_calculate(self, gap_coord):
        self.gap_center = (gap_coord[0] + gap_coord[-1]) // 2

class MyGame(ar.Window):
    def __init__(self, w, h, t):
        super().__init__(w, h, t)

        self.player_list = None
        self.distance_list = None
        self.leader_list = None
        self.brain_list = None
        self.wall = None
        self.pause = False
        self.current_state = START_STATE
        self.generation_count = 0
        self.hit_list = None

    def setup(self):
        ar.set_background_color(ar.color.AMAZON)

        self.generation_count += 1

        self.leader_list = []
        self.distance_list = []
        self.player_list = []
        self.hit_list = []

        for _ in range(INDIVIDUAL_Q):
            player = Ball(50, H / 2, 15)
            self.player_list.append(player)

        if self.current_state == START_STATE:
            self.brain_list = []
            for i in range(INDIVIDUAL_Q):
                brain = NeuralNetwork(NEURON_LAYERS, INIT_WEIGHT_MAX)
                self.brain_list.append(brain)
            self.current_state = GAME_RUNNING

        self.wall = Wall()
        self.wall.setup()

    def on_draw(self):
        ar.start_render()
        
        for player in self.player_list:
            if not player.stop:
                player.draw()
        
        self.wall.draw()

        if self.current_state == GAME_RUNNING:
            output = f'Generation: {self.generation_count}'
            ar.draw_text(output, W - 200, 200, ar.color.WHITE, 14, font_name='Arial')
            
            #output = f'Lenght HitList: {len(self.hit_list)}'
            #ar.draw_text(output, W - 200, 200, ar.color.WHITE, 14, font_name='Arial')

            for index, player in enumerate(self.player_list):
                output = f'Distance #{index}: {player.distance_traveled}'
                ar.draw_text(output, W - 200, 20 + 20 * index, ar.color.WHITE, 14, font_name='Arial')

    def update(self, delta_time):
        if self.pause:
            return
        if self.current_state == GAME_RUNNING:
            self.wall.update()

            for idx, player in enumerate(self.player_list):
                if not player.stop:
                    for brick in self.wall.wall_list:
                        if self.check_for_collision(player, brick):
                            player.stop = True
                            self.hit_list.append(player)

                    if player.position_y < 0 or player.position_y > H:
                        player.stop = True
                        self.hit_list.append(player)

                    dist_x = self.wall.wall_center_x - player.position_x
                    dist_y = self.wall.gap_center - player.position_y
                    input_data = [dist_x / W, dist_y / H]
                    player.distance_to_wall = dist_x
                    if self.brain_list[idx](input_data) >= 0.5:
                        player.change_y = JUMP_SPEED
                    else:
                        player.change_y = -GRAVITY
                    player.update()

                else:
                    continue

            if len(self.hit_list) >= len(self.player_list):
                self.current_state = GAME_OVER
        
        if self.current_state == GAME_OVER:
            for player in self.player_list:
                result_distance = player.distance_traveled - player.distance_to_wall
                self.distance_list.append(result_distance)

            self.brain_list = selection(self.distance_list, 
                                        self.brain_list, 
                                        LEADER_Q,
                                        INDIVIDUAL_Q, 
                                        CROSS_PRECEPT, 
                                        MUTATION_PRECEPT, 
                                        LEARN_RATE, MUT_MAX)

            for player in self.player_list:
                player.stop = True
            self.current_state = GAME_RUNNING
            self.setup()

    def check_for_collision(self, subject1, subject2) -> bool:
        collision_radius_sum_x = subject1.radius + subject2.width / 2
        collision_radius_sum_y = subject1.radius + subject2.height / 2

        diff_x = subject1.position_x - subject2.position[0]
        diff_x2 = diff_x * diff_x

        if diff_x2 > collision_radius_sum_x * collision_radius_sum_x:
            return False

        diff_y = subject1.position_y - subject2.position[1]
        diff_y2 = diff_y * diff_y
        if diff_y2 > collision_radius_sum_y * collision_radius_sum_y:
            return False

        # distance = math.sqrt(diff_x * diff_x + diff_y * diff_y)
        # if distance > collision_radius_sum:
        #     return False

        distance = diff_x2 + diff_y2
        if distance > collision_radius_sum_x * collision_radius_sum_y:
            return False

        return True

def main():
    window = MyGame(W, H, 'Neuro game')
    window.setup()
    ar.run()

main()