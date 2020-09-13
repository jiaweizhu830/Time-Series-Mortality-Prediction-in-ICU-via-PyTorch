import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
from sklearn.metrics import *

# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plt.plot(train_losses)
	plt.plot(valid_losses)
	plt.legend(['Training losses', 'Validation losses'], loc='upper right')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.savefig('loss'+'.png')
	plt.clf()
	plt.cla()
	plt.close()

	plt.plot(train_accuracies)
	plt.plot(valid_accuracies)
	plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.savefig('accuracy'+'.png')
	plt.clf()
	plt.cla()
	plt.close()

def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# Compute confusion matrix
	results = np.asarray(results)
	y_true = results[:,0]
	y_pred = results[:,1]
	cm = confusion_matrix(y_true, y_pred)

	classes = class_names
	#classes = classes[unique_labels(y_true, y_pred)]
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots()
	im = ax.imshow(cm, vmin=0, vmax=1.0, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]),
		yticks=np.arange(cm.shape[0]),
		# ... and label them with the respective list entries
		xticklabels=classes, yticklabels=classes,
		ylabel='True label',
		xlabel='Predicted label')
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")
	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
				ha="center", va="center",
				color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.savefig('cm'+'.png')
	plt.clf()
	plt.cla()
	plt.close()
	pass

def plot_roc_curves(y_true, y_pred):

	fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
	au = roc_auc_score(y_true, y_pred)
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % au)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.savefig('ROC_curve'+'.png')
	plt.clf()
	plt.cla()
	plt.close()