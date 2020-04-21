
## k-NN comparisson

Comparisson between k-NN's algorithms
1. k-NN (Cover, Hart)
2. WKNN (S. Dudani, The Distance-Weighted k-Nearest-Neighbor Rule, 1976)
3. LMKNN (Mitani, Y., Hamamoto, Y., A local mean-based nonparametric classifier, 2006
4. PNN (Zeng, Yong; Yang, Yupu; Zhao, Liang, Pseudo nearest neighbor rule for pattern classification, 2009)
5. MKNN (Liu, Huawen; Zhang, Shichao, Noisy data elimination using mutual k-nearest neighbor for classification mining, 2012)

## Description

- support multiple ***k*** values, e.g. k = [3, 5, 7]
- ***UCI*** datasets included
- validation with ***10-fold cross validation***
- evaluation using ***error rate*** and ***f-measure***

## Prerequisites

Items should be ready:
- [x] Python 2.7
- [x] sklearn
- [x] pandas
- [x] scipy
- [x] numpy
- [x] statistics

---

## How to run program
To run the program, please following this steps:
1. Copy all files in `dataset/data` to python installation directory, e.g: `C:/Python27/Lib/site-packages/sklearn/datasets/data`
2. Copy all files in `dataset/descr` to python installation directory e.g.  `C:/Python27/Lib/site-packages/sklearn/datasets/descr`
3. Open Python installation directory
4. Edit file `C:/Python27/Lib/site-packages/sklearn/datasets/__init__.py`
   Add command
   `from .base import load_[dataset_name]` and
   `'load_[dataset_name]'` to `__all__`
	e.g:
	`from .base import load_thyroid`
	then in `__all__` add command `'load_thyroid'`
	
5. Edit file `C:/Python27/Lib/site-packages/sklearn/datasets/base.py`
   Create a function `load_[dataset_name]`
   e.g:
   ```
	   def load_thyroid(return_X_y=False):
   	module_path = dirname(__file__)
    	data, target, target_names = load_data(module_path, 'thyroid.csv')

    	with open(join(module_path, 'descr', 'thyroid.rst')) as rst_file:
		     fdescr = rst_file.read()

    	if return_X_y:
        	     return data, target

    	return Bunch(data=data, target=target,
                target_names=target_names,
                DESCR=fdescr,
                feature_names=['T3-resin', 'Serum thyroxin', 'triiodothyronine', 'TSH', 'Maximal absolute', 'class'])
	```		
			
	Description:
	- `thyroid.srt` is a dataset description. You can write down on it all about dataset.
	- `thyroid.csv` is a dataset file with first row content is
	  215 | 5 | normal | hyper | hypo means 215 data, 5 attributes, class[0] = normal, class[1] = hyper, class[2] = hypo
	- You may add your own dataset by following this steps:
	  1. save your dataset in .csv file then put in `../data`
	  2. create description file in `.srt` then put in `../descr`
	  3. add command to load new dataset in `__init__.py`
	  4. add command to load new dataset in `base.py`
	  5. to check it's work or not, you can run:
	  ```
	  from sklearn import datasets
	  data = datasets.load_thyroid()
	  print data
		```
6. Open `main.py` 
   set to 1 in `selected_datasets` which dataset you want to use
   set 1 on `classifier` which claasifier algorithms you want to compare
   set `k_values`, can be single or multiple value
7. Open command prompt/terminal, navigate to this repository, then run
`python main.py`
8. Output program will be write on `logs/experiment_log.txt`. 

***Fell free to ask me at tofa.zakie@gmail.com*** 
	
	
