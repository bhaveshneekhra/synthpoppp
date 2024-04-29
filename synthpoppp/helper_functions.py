import numpy as np

TINY = 1e-20
LN2 = np.log(2)
# R0 = 0.01

R0 = 10

CONST = LN2/R0

###Helper functions for data cleaning
def try_convert(value, default, *types):
	for t in types:
		try:
			return t(value)
		except (ValueError, TypeError):
			continue
	return default

"""
def get_probabilistic_place_assignment_batch_zipf(individuals_geocode, places_geocode):
	individuals_geocode_sq_sum = np.power(np.linalg.norm(individuals_geocode, axis=1, keepdims=True),2)
	workplaces_geocode_sq_sum = np.power(np.linalg.norm(places_geocode, axis=1, keepdims=True),2)
	inverse_distances = np.power(0.1, -(individuals_geocode_sq_sum+workplaces_geocode_sq_sum.T-2*np.matmul(individuals_geocode, places_geocode.T)+TINY))
	inverse_distances = inverse_distances/inverse_distances.sum(axis=1,keepdims=True)
	return (inverse_distances.cumsum(axis=1)>np.random.uniform(size=(individuals_geocode.shape[0],1))).argmax(axis=-1)
"""

def get_probabilistic_place_assignment_batch_zipf(individuals_geocode, places_geocode):
	place_geocodes_norm_sq = np.power(places_geocode.T, 2).sum(axis=0, keepdims=True)
	individuals_geocodes_norm_sq = np.power(individuals_geocode, 2).sum(axis=1, keepdims=True)
	distances_km = 111 * np.sqrt(place_geocodes_norm_sq + individuals_geocodes_norm_sq - 2 * np.matmul(individuals_geocode, places_geocode.T) + TINY)
	f = CONST*np.power(2, -distances_km/R0)
	return ((f/f.sum(axis=-1, keepdims=True)).cumsum(axis=-1)>np.random.uniform(size=(individuals_geocode.shape[0],1))).argmax(axis=-1)

def get_probabilistic_place_assignment_batch_default(individuals_geocode, places_geocode):
	individuals_geocode_sq_sum = np.power(np.linalg.norm(individuals_geocode, axis=1, keepdims=True),2)
	workplaces_geocode_sq_sum = np.power(np.linalg.norm(places_geocode, axis=1, keepdims=True),2)
	inverse_distances = 1/(individuals_geocode_sq_sum+workplaces_geocode_sq_sum.T-2*np.matmul(individuals_geocode, places_geocode.T)+TINY)
	inverse_distances = inverse_distances/inverse_distances.sum(axis=1,keepdims=True)
	inverse_distances = inverse_distances.cumsum(axis=-1)
	uniform_sample = np.random.uniform(size=(individuals_geocode.shape[0],1))
	return (inverse_distances>uniform_sample).argmax(axis=-1)

def get_probabilistic_place_assignment(individuals_geocode, places_geocode_df, batch_size=10000, p_type='default'):
# 	n_batches = int(np.ceil(len(individuals_geocode)/batch_size))
	
# 	batch_wise_indicies = []

# 	# batch_wise_indicies = np.zeros(len(adults), dtype=int)

# 	if(p_type=='default'):
# 		get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_default
# # 		pass
# 	elif(p_type=='zipf'):
# 		get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_zipf
# 	else:
# 		# get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_default
# 		pass

# 	# places_geocode = places_geocode_df[['W_Lat', 'W_Lon']].values

# 	for batch_counter in tqdm.tqdm(range(n_batches)):
		
# 		begin_index = (batch_counter)*batch_size
# 		end_index = (batch_counter+1)*batch_size

# 		batch_wise_index = get_probabilistic_place_assignment_batch(individuals_geocode[begin_index:end_index], places_geocode_df)
        
# # 		print(batch_wise_index)
# # 		print(len(batch_wise_index))
# # 		break

# 		batch_wise_indicies[(batch_counter)*batch_size:(batch_counter+1)*batch_size] = places_geocode_df.loc[batch_wise_index].index

# 		wps, wps_count = np.unique(batch_wise_index, return_counts=True)

# # 		print(wps, wps_count)
        
# # 		print(places_geocode_df.loc[wps])
# # 		break

# 		wkplace_index = places_geocode_df.loc[wps].index

# 		workplaces.loc[wkplace_index, 'Size'] -= wps_count
		
# 		places_geocode_df = workplaces[workplaces['Size'] > 0][['W_Lat', 'W_Lon']]

# 		remaining_wkplace_count = len(workplaces[workplaces['Size'] > 0])
		
# 		if batch_counter % 10000 == 0:
# 			print("Remaining places which can be assigned: ", remaining_wkplace_count)
# 		# # print(batch_wise_index.min(), batch_wise_index.max())
		

# 		if remaining_wkplace_count == 0:
# 			print("No more places remaining to be assigned!")
# 			print("Was assigning for the index: ", begin_index, "and ", end_index)
# 			break
		
		
		
# # 		batch_wise_indicies.append(batch_wise_index)

# 	return (batch_wise_indicies)



	n_batches = int(np.ceil(len(individuals_geocode)/batch_size))
	batch_wise_indicies = []

	if(p_type=='default'):
		get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_default
	elif(p_type=='zipf'):
		get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_zipf
	else:
		get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_default

	for batch_counter in range(n_batches):
		batch_wise_indicies.append(get_probabilistic_place_assignment_batch(individuals_geocode[(batch_counter)*batch_size:(batch_counter+1)*batch_size], places_geocode_df))
	return np.concatenate(batch_wise_indicies)
