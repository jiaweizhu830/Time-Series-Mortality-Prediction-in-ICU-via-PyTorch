# The framework is based on homework 5 of CSE6250 2019 spring at Georgia Tech.
# The codes are implemented by ourself.
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix, plot_roc_curves
from mydatasets import calculate_num_features, VisitSequenceWithLabelDataset, visit_collate_fn
from mymodels import LSTM

torch.manual_seed(0)
if torch.cuda.is_available():
	torch.cuda.manual_seed(0)

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = "../data/features/train/mortality.seqs.train"
PATH_TRAIN_LABELS = "../data/features/train/mortality.labels.train"
#PATH_TRAIN_VALUES = "../data/processed/mortality.values.train"
PATH_VALID_SEQS = "../data/features/validation/mortality.seqs.validation"
PATH_VALID_LABELS = "../data/features/validation/mortality.labels.validation"
#PATH_VALID_VALUES = "../data/processed/mortality.values.validation"
PATH_TEST_SEQS = "../data/features/test/mortality.seqs.test"
PATH_TEST_LABELS = "../data/features/test/mortality.labels.test"
#PATH_TEST_VALUES = "../data/processed/mortality.values.test"
PATH_OUTPUT = "../output/"

NUM_EPOCHS = 1
BATCH_SIZE = 64
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0

# Data loading
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
#train_values = pickle.load(open(PATH_TRAIN_VALUES, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
#valid_values = pickle.load(open(PATH_VALID_VALUES, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))
#test_values = pickle.load(open(PATH_TEST_VALUES, 'rb'))
print('===> done Loading')
num_features = calculate_num_features(train_seqs)
print(num_features)
#print(len(train_seqs[1]))

train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels, num_features)
valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)
test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features)
print('===> done datasets')

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)


model = LSTM(num_features)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=0.004, weight_decay = 0.0007)
#optimizer = optim.SGD(model.parameters(), lr=0.0004)

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
model.to(device)
criterion.to(device)

optimizer = optim.Adam(model.parameters(),lr=0.007, weight_decay = 0.0004)
best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, "LSTM.pth"))

print(best_val_acc)	

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

best_model = torch.load(os.path.join(PATH_OUTPUT, "LSTM.pth"))


# TODO: Complete predict_mortality
def predict_mortality(model, device, data_loader):
	model.eval()

	results = []

	model.eval()

	with torch.no_grad():
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)

			m = nn.Sigmoid()

			y_pred = m(output.detach().to('cpu'))[0][1].numpy().tolist()
			results.append(y_pred)
	

	probas = results
	return probas

test_prob = predict_mortality(best_model, device, test_loader)
test_label = pickle.load(open(PATH_VALID_LABELS, "rb"))

plot_roc_curves(test_label, test_prob)

test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion)
class_names = ['Alive', 'Dead']
plot_confusion_matrix(test_results, class_names)
