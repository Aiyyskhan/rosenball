import numpy as np
import copy
from operator import itemgetter

# функция отбора
def selection(result_list: list, 
				chromosome_list: list,
				leader_q: int, 
				indiv_q: int, 
				cross_precept: list, 
				mut_precept: list, 
				learn_rate: float, 
				mut_max: float) -> list:
	leader_list = None
	# сортировка по результатам
	result_list, chromosome_list = zip(*sorted(zip(result_list, chromosome_list), key=itemgetter(0), reverse=True))
	#result_list, chromosome_list = [list(i) for i in zip(*sorted(zip(result_list, chromosome_list), key=itemgetter(0), reverse=True))]
	result_list = list(result_list)
	chromosome_list = list(chromosome_list)

	# отбор лидеров
	leader_list = chromosome_list[:leader_q]
	chromosome_list.clear()

	# скрещивание лидеров и генерация новых особей
	crossover(chromosome_list, leader_list, indiv_q, cross_precept)

	# мутация некоторых особей
	#if (i % MUT_INTERVAL) == 0: 
	mutation(chromosome_list, mut_precept, learn_rate, mut_max)

	return chromosome_list

# функция кроссинговера
def crossover(indiv_list: list, lead_list: list, indiv_q: int, precept_list: list):
	def hybridization(ind_syn: 'array', lead_syn: 'array', ratio_percent: int):
		ind_syn_rav = ind_syn.ravel()
		lead_syn_rav = lead_syn.ravel()

		len_ind_syn = len(ind_syn_rav)
		permut_index = np.random.permutation(len_ind_syn)
		for i in range(int(round((ratio_percent/100)*len_ind_syn, 0))):
			ind_syn_rav[permut_index[i]] = lead_syn_rav[permut_index[i]]
				
	for i in range(indiv_q):
		individ = copy.deepcopy(lead_list[0])
		indiv_list.append(individ)
	
	for precept in precept_list:
		for i in range(len(indiv_list[precept[0]].syn_list)):
			hybridization(indiv_list[precept[0]].syn_list[i], lead_list[precept[1][0]].syn_list[i], precept[1][1])

# функция мутации
def mutation(indiv_list: list, precept_list: list, learn_rate: float, mut_max: float):
	def mutate(ind_syn: 'array', ratio_percent: int):
		ind_syn_rav = ind_syn.ravel()

		len_ind_syn = len(ind_syn_rav)
		permut_index = np.random.permutation(len_ind_syn)
		for i in range(int(round((ratio_percent/100)*len_ind_syn, 0))):
			ind_syn_rav[permut_index[i]] += learn_rate * (2 * (np.random.random() - 0.5) * mut_max)
	
	for precept in precept_list:
		for i in range(len(indiv_list[precept[0]].syn_list)):
			mutate(indiv_list[precept[0]].syn_list[i], precept[1])