from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import statistics
import pandas as pd
from knn import *

# initiate datasets
all_datasets = [None] * 10

all_datasets[0] = datasets.load_hepatitis() 
all_datasets[1] = datasets.load_wine()
all_datasets[2] = datasets.load_sonar()
all_datasets[3] = datasets.load_seeds()
all_datasets[4] = datasets.load_glass()
all_datasets[5] = datasets.load_thyroid()
all_datasets[6] = datasets.load_haberman()
all_datasets[7] = datasets.load_ecoli()
all_datasets[8] = datasets.load_balance()
all_datasets[9] = datasets.load_vowel()

datasets_name     = ['1. HEPATITIS', '2. WINE', '3. SONAR', '4. SEEDS', '5. GLASS', '6. THYROID', '7. HABERMAN', '8. E. COLI', '9. BALANCE', '10. VOWEL']
selected_datasets = [0,               0,         0,          0,          0,          1,            0,             0,            0,            0]

# methods     knn,  wknn, lmknn, mknn, pnn       
classifier = [  1,     1,     1,    1,   1]

# k value can be single or multiple
k_values = [3, 5, 7]
n_fold = 10

# set log file
log_file = 'logs/experiment_log.txt'

# loop all datasets
for i in range(len(all_datasets)):
	if(selected_datasets[i] == 0):
		continue

	with open(log_file, "a") as text_file:
		text_file.write("%s" % datasets_name[i])
		text_file.write("\n")
		text_file.write("************************************")
		text_file.write("\n")

	# set dataset
	dataset = all_datasets[i]

	# loop for each k values
	for k in k_values:
		# set initial error rate
		err_knn 	= []
		err_wknn 	= []
		err_lmknn 	= []
		err_mknn 	= []
		err_pnn 	= []

		# set intial fscore
		f1_knn 			= []
		f1_wknn 		= []
		f1_lmknn 		= []
		f1_mknn 		= []
		f1_pnn 			= []
		
        # write log
		with open(log_file, "a") as text_file:
			text_file.write("k = %s" % k)
			text_file.write("\n")

		# split data
		skf = KFold(n_splits=n_fold, shuffle=True)
		split = skf.split(dataset.data, dataset.target)

		date_row 	= []
		date_row_f 	= []
		cols 		= []
		means 		= []
		means_f	    = []
		i = 1
		# SPLIT - 10 FOLD CROSS VALIDATION
		for train_idx, test_idx in split:
			print '-------------------------'
			print 'Fold: ', i
			print 'Data Train: ', len(train_idx)
			print 'Data Test: ', len(test_idx)

			i = i + 1
			X_train	= dataset.data[train_idx]
			X_test 	= dataset.data[test_idx]
			y_train = dataset.target[train_idx]
			y_test 	= dataset.target[test_idx]

			row		= []
			row_f	= []

			# classifier
			if(classifier[0]):
				e_knn, f_knn = knn(k, dataset.data[train_idx], dataset.data[test_idx], dataset.target[train_idx], dataset.target[test_idx])
				err_knn = err_knn + [e_knn]
				row = row + [e_knn]

				f1_knn  = f1_knn + [f_knn]
				row_f = row_f + [f_knn]
				print 'knn finished'

			if(classifier[1]):
				e_wknn, f_wknn = wknn(k, dataset.data[train_idx], dataset.data[test_idx], dataset.target[train_idx], dataset.target[test_idx])
				err_wknn = err_wknn + [e_wknn]
				row = row + [e_wknn]

				f1_wknn  = f1_wknn + [f_wknn]
				row_f = row_f + [f_wknn]
				print 'wknn finished'

			if(classifier[2]):
				e_lmknn, f_lmknn = lmknn(k, dataset.data[train_idx], dataset.data[test_idx], dataset.target[train_idx], dataset.target[test_idx])
				err_lmknn = err_lmknn + [e_lmknn]
				row = row + [e_lmknn]

				f1_lmknn  = f1_lmknn + [f_lmknn]
				row_f = row_f + [f_lmknn]
				print 'lmknn finished'

			if(classifier[3]):
				e_mknn, f_mknn = mknn(k, dataset.data[train_idx], dataset.data[test_idx], dataset.target[train_idx], dataset.target[test_idx])
				err_mknn = err_mknn + [e_mknn]
				row = row + [e_mknn]

				f1_mknn  = f1_mknn + [f_mknn]
				row_f = row_f + [f_mknn]
				print 'mknn finished'

			if(classifier[4]):
				e_pnn, f_pnn   = pnn(k, dataset.data[train_idx], dataset.data[test_idx], dataset.target[train_idx], dataset.target[test_idx])
				err_pnn = err_pnn + [e_pnn]
				row = row + [e_pnn]

				f1_pnn  = f1_pnn + [f_pnn]
				row_f = row_f + [f_pnn]
				print 'pnn finished'


			date_row.append(row)
			date_row_f.append(row_f)
			print ''

		# MEANS
		if(classifier[0]):
			cols = cols + ['knn']
			means = means + [statistics.mean(err_knn)]
			means_f = means_f + [statistics.mean(f1_knn)]

		if(classifier[1]):
			cols = cols + ['wknn']
			means = means + [statistics.mean(err_wknn)]
			means_f = means_f + [statistics.mean(f1_wknn)]

		if(classifier[2]):
			cols = cols + ['lmknn']
			means = means + [statistics.mean(err_lmknn)]
			means_f = means_f + [statistics.mean(f1_lmknn)]

		if(classifier[3]):
			cols = cols + ['mknn']
			means = means + [statistics.mean(err_mknn)]
			means_f = means_f + [statistics.mean(f1_mknn)]

		if(classifier[4]):
			cols = cols + ['pnn']
			means = means + [statistics.mean(err_pnn)]
			means_f = means_f + [statistics.mean(f1_pnn)]


		date_row.append(means)
		date_row_pd = pd.DataFrame(data = date_row, columns=cols)

		date_row_f.append(means_f)
		date_row_pd_f = pd.DataFrame(data = date_row_f, columns=cols)


		with open(log_file, "a") as text_file:
			text_file.write("Error rate \n")
			text_file.write("%s" % date_row_pd)
			text_file.write("\n")
			text_file.write("\n")
			text_file.write("F-measure \n")
			text_file.write("%s" % date_row_pd_f)
			text_file.write("\n")
			text_file.write("\n")