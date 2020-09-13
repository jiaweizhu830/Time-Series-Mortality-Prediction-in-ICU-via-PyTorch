import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	flat_list = []
	for patient in seqs:
		for sublist in patient:
			for (item,_) in sublist:
				flat_list.append(item)
	myset = set(flat_list)
	mynewlist = list(myset)
#	flat_list =list( map(int, flat_list))
	return  max(mynewlist)+1 #len(mynewlist)


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		self.seqs = []  # replace this with your implementation.
		for patient in seqs:
			rows = len(patient)
			I = []
			J = []
			V = []
			i = 0
			for sublist in patient:
				columns = len(sublist)
				x = [i for (i,j) in sublist]
				y = [j for (i,j) in sublist]
				I = I + [i]*columns
				i=i+1
				J = J + x
				V = V + y
			B = sparse.coo_matrix((V,(I,J)),shape=(rows,num_features)).tocsr().toarray()
			self.seqs.append(B)
	def __len__(self):
	 	return len(self.labels)

	def __getitem__(self, index):
	 	# returns will be wrapped as List of Tensor(s) by DataLoader
	 	return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence
	batch.sort(key=lambda x: len(x[0]), reverse=True)
	batch_size = len(batch)
	max_length = len(batch[0][0])
	num_features = len(batch[0][0][0])
	labels = []
	sizes = []
	targets = torch.zeros(batch_size, max_length, num_features).float()
	for i, x in enumerate(batch):
		end = len(x[0])
		labels.append(x[1])
		sizes.append(len(x[0]))
		targets[i, :end] = torch.from_numpy(x[0].astype('float32'))
	labels = np.asarray(labels)
	length = np.asarray(sizes)
	seqs_tensor = targets
	lengths_tensor = torch.from_numpy(length.astype('int64'))
	labels_tensor = torch.from_numpy(labels.astype('int64'))

	return (seqs_tensor, lengths_tensor), labels_tensor
