Author          : Baolin Wang, Jiawei Zhu, Shimiao Zhang
Created         : April 27, 2019
Last Modified   : April 27, 2019

Affiliation     : Georgia Institute of Technology
==========================================================================================================

==============================
Execution

1. python dataProcess.py (see section "Data Preprocessing")
2. sbt run compile (see section "Feature construction program")
3. change names of the generated csv files (see section "Feature construction program")
4. python feature.py
5. python train_lstm.py


==============================
Description

------------------------------
Data Preprocessing
------------------------------
This program performs preprocessing of the datasets used in this project (ADMISSIONS.csv, DIAGNOSES_ICD.csv, ICUSTAYS.csv, LABEVENTS.csv, PATIENTS.csv, PRESCRIPTIONS.csv). It splits the raw datasets into train, validation and test subsets, and put them in the [train | valid | test] folders. The output can be found in https://drive.google.com/file/d/1HEPSv2LY8mLoSRyWbTbtMEtEdGHng2nN/view?usp=sharing. Before running the step 2 above (i.e. sbt run compile), these csv files should be put in the corresponding [train | valid | test] sub-folders in the current directory (which are empty now).

dataProcess.py
main program for data preprocessing

------------------------------
Feature construction program
------------------------------
This program reads preprocessed data, constructs feature map based on diagnostic codes, medications, lab results, 
and length of ICU stays, and map the constructed feature ids to the data. It outputs the pickle files for LSTM 
model program. This program can be run in the image provided by CSE 6250 course in docker. The "sbt run compile"
command should be run at the current folder.

src/main/scala/edu/gatech/cse6250/model/models.scala
functions to set the RDD format

src/main/scala/edu/gatech/cse6250/helper/CSVHelper.scala
src/main/scala/edu/gatech/cse6250/helper/SparkHelper.scala
functions to help to load csv files

src/main/scala/edu/gatech/cse6250/main/Main.scala
functions to load data from csv files produced from preprocessing and construct features

src/main/scala/edu/gatech/cse6250/featureconstruct/FeatureConstructor.scala
functions to construct feature map, map features to the datasets, and output resultant RDDs to csv files

The csv files can be found in folders: featureDF_train, featureDF_valid, featureDF_test, patientIDLabels_train, patientIDLabels_valid, and patientIDLabels_test 
In featureDF_train folder, names of csv files should be changed to train1.csv, train2.csv, ..., train8.csv from top to bottom
In featureDF_valid folder, names of csv files should be changed to valid1.csv, valid2.csv, ..., valid5.csv from top to bottom
In featureDF_test folder, names of csv files should be changed to test1.csv, test2.csv, ..., test5.csv from top to bottom
In patientIDLabels_train, name of the csv file should be changed to pTrain1.csv
In patientIDLabels_valid, name of the csv file should be changed to pValid1.csv
In patientIDLabels_test, name of the csv file should be changed to pTest1.csv

feature.py
functions to convert csv files produced from featureConstructor.scala to sequence data, list of patient ids, and list of labels, and output results to pickle files
The output pickle files can be found in folders: ../data/features/[train | test | validation]


------------------------------
LSTM model program
------------------------------
This program uses temporal electronic healthcare records data with a Recurrent 
neural network (RNN) model called Long short-term memory (LSTM) to do prediction 
on mortality of patients in intensive care units. This program was written in python.

mydatasets.py
Utility functions that are use to create custom dataset (3D Tensor).

utils.py
Utility functions that are use to train and evaluation the model.

plots.py
Utility functions that are use to generate figures to show the results and predictions

mymodels.py
Specify the LSTM model.

train_lstm.py
main code to run this program.


==============================
Reference

We referred to Homework 4 and Homework 5 of CSE6250 2019 Spring @Georgia Tech when we develop this project. Only a small part of the codes are directly coming from their skeleton, most codes are implemented by ourselves.