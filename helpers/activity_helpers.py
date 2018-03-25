import numpy as np
import ipdb

def calculate_entropies(activity_list, labels):

	'''
	activity_list: (list) Each element is an array of activities for a single layer
		       Arrays are num_samples x height x width x channels
	'''

	for activities in activity_list:
		# Positive Entropy
			positive_activities = activity[np.where(labels == 1), :]
			positive_entropy = layer_entropy(positive_activities)	
		# Negative Entropy
			negative_activities = activity[np.where(labels == 0), :]
			negative_entropy = layer_entropy(negative_activities)
		# All Entropy
			all_entropy = layer_entropy(activities)

	return positive_entropy, negative_entropy, all_entropy

def layer_entropy(activies, num_bins=20):
	
	num_samples = np.shape(activities)[0]
	activities = np.resape(activities, [num_samples, -1])
	num_units = np.shape(activities)[1]

	min_activity = np.min(activities)
	max_activity = np.max(activities)
	
	all_distributions = []
	for uu in range(num_units):
		activity = activities[:,uu]
		distribution, _ = np.histogram(activity, num_bins, range=(min_activity,max_activity))
		all_distributions.append((1/float(num_samples))*distribution)
	
	all_distributions = np.array(all_distributions) 
	unit_entropies = entropy(all_distributions)
	
	return np.mean(unit_entropies)

def entropy(array, axis=1):

	summands = array*np.log(array)
	summands[np.isnan(summands)] = 0
	
	return -1*np.sum(summands, axis)

def online_covariance_step(array, current_mean, current_covariance, num_samples, weight=1):
    array = np.squeeze(np.reshape(array, -1))
    delta = weight*array - np.reshape(current_mean, -1)
    print(current_covariance.shape)
    print(np.outer(delta, np.transpose(delta)).shape)
    current_covariance  += (1.0/ float(num_samples))*np.outer(delta, np.transpose(delta))
    
    return current_covariance
