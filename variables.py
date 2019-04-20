# количество нейронов в каждом слое
NEURON_LAYERS = [2, 3, 1]

# гиперпараметры
LEARN_RATE = 1
INIT_WEIGHT_MAX = 0.5

# numbers of individuals
INDIVIDUAL_Q = 8

# numbers of leaders
LEADER_Q = 5

# mutation range
MUT_MAX = 1

# precept crossover and mutation
CROSS_PRECEPT = [#[0,[1,10]], 
                [1,[1,70]], 
                [2,[1,90]],
                [2,[3,20]], 
                [3,[1,100]], 
                [3,[2,50]], 
                [4,[2,100]], 
                [4,[4,50]],
				[5,[1,100]],
				[5,[4,30]],
				[6,[4,70]],
				[7,[3,100]],
				[7,[2,30]]]

MUTATION_PRECEPT = [[1,30],
                    [2,100],
                    [3,30],
                    [4,100],
					[5,70],
					[6,100],
					[7,100]]