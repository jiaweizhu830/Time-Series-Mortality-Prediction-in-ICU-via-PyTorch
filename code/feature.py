import pandas as pd
import pickle
import os

# specify output path
PATH_TRAIN = "../data/features/train/"
PATH_VALID = "../data/features/validation/"
PATH_TEST = "../data/features/test/"

# set which dataset ("train" or "validation" or "test") to run
setType = "test"

# load data
if setType == "train":
    print("load training data")
    df1 = pd.read_csv("featureDF_train/train1.csv", header=None)
    df1.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df2 = pd.read_csv("featureDF_train/train2.csv", header=None)
    df2.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df3 = pd.read_csv("featureDF_train/train3.csv", header=None)
    df3.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df4 = pd.read_csv("featureDF_train/train4.csv", header=None)
    df4.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df5 = pd.read_csv("featureDF_train/train5.csv", header=None)
    df5.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df6 = pd.read_csv("featureDF_train/train6.csv", header=None)
    df6.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df7 = pd.read_csv("featureDF_train/train7.csv", header=None)
    df7.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df8 = pd.read_csv("featureDF_train/train8.csv", header=None)
    df8.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    frame = [df1, df2, df3, df4, df5, df6, df7, df8]

    patientDF = pd.read_csv("patientIDLabels_train/pTrain1.csv")
    patientDF.columns = ["patientID", "IS_DEAD"]

elif setType == "validation":
    print("load validation data")
    df1 = pd.read_csv("featureDF_valid/valid1.csv", header=None)
    df1.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df2 = pd.read_csv("featureDF_valid/valid2.csv", header=None)
    df2.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df3 = pd.read_csv("featureDF_valid/valid3.csv", header=None)
    df3.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df4 = pd.read_csv("featureDF_valid/valid4.csv", header=None)
    df4.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df5 = pd.read_csv("featureDF_valid/valid5.csv", header=None)
    df5.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    frame = [df1, df2, df3, df4, df5]

    patientDF = pd.read_csv("patientIDLabels_valid/pValid1.csv")
    patientDF.columns = ["patientID", "IS_DEAD"]

elif setType == "test":
    print("load test data")
    df1 = pd.read_csv("featureDF_test/test1.csv", header=None)
    df1.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df2 = pd.read_csv("featureDF_test/test2.csv", header=None)
    df2.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df3 = pd.read_csv("featureDF_test/test3.csv", header=None)
    df3.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df4 = pd.read_csv("featureDF_test/test4.csv", header=None)
    df4.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    df5 = pd.read_csv("featureDF_test/test5.csv", header=None)
    df5.columns = ["patientID", "admitTime", "featureID", "featureValue"]
    frame = [df1, df2, df3, df4, df5]

    patientDF = pd.read_csv("patientIDLabels_test/pTest1.csv")
    patientDF.columns = ["patientID", "IS_DEAD"]

else:
    print("wrong dataset type")

print("concat DF")
# concatenate dataframes and modify data types in the dataframe
featureDF = pd.concat(frame, axis=0)
featureDF.patientID = featureDF.patientID.astype('int64')
featureDF.admitTime = featureDF.admitTime.astype('int64')
featureDF.featureID = featureDF.featureID.astype('int64')
featureDF.featureValue = featureDF.featureValue.astype('float')

patientDF.patientID = patientDF.patientID.astype('int64')
patientDF.IS_DEAD = patientDF.IS_DEAD.astype('int')

print("append to labels")
# convert to list of patient ids and list of labels
patientDFSort = patientDF.sort_values(by=['patientID'], ascending=True).copy()
patient_ids = list(patientDFSort.patientID.values)
labels = list(patientDFSort.IS_DEAD.values)

print("group by feature DF")
# filter feature dataframe based on the patient dataframe
idFeatureDF = featureDF.set_index(featureDF.patientID).index
idPatientDF = patientDF.set_index(patientDF.patientID).index
featureFiltered = featureDF[idFeatureDF.isin(idPatientDF)].copy()
featureFiltered['features'] = list(zip(featureFiltered.featureID, featureFiltered.featureValue))
# sort feature dataframe by patient id and admit time, and group by patient id and admit time
featureSorted = featureFiltered.sort_values(by=['patientID', 'admitTime'], ascending=[True, True]).copy()
featureGB = featureSorted.groupby(['patientID', 'admitTime']).features.apply(list)

print("append to seqs and ids")
# initialize
patient_lst = []
seq_data = []
date_lst = []
dummy_id = -1
dummy_date = -1
count = 0
len_featureGB = featureGB.shape[0]

# Create sequence data like HW5
# The sequence data list contains a list of patient lists
# Each patient list contains a list of date lists
# Each date list contains a list of feature tuples
# Each feature tuple includes feature id, and feature value
#
# for example,
# datei = [(feature_id, feature_value), (feature_id, feature_value), ...]
# patienti = [date1, date2, ..., daten]
# sequence data = [patient1, patient2, ...., patientn]

for idx, row in featureGB.iteritems():
    if count == 0:
        dummy_date = idx[1]
        dummy_id = idx[0]

    if dummy_id != idx[0]:
        patient_lst.append(date_lst)
        seq_data.append(patient_lst)
        date_lst = []
        patient_lst = []

        dummy_date = idx[1]
        dummy_id = idx[0]

    if (dummy_id == idx[0]) and (dummy_date == idx[1]):
        if len(date_lst) == 0:
            date_lst.extend(row)
        else:
            date_lst.append(row)

    if (dummy_id == idx[0]) and (dummy_date != idx[1]):
        patient_lst.append(date_lst)
        date_lst = []
        date_lst.extend(row)
        dummy_date = idx[1]

    count += 1
    dummy_date = idx[1]
    dummy_id = idx[0]

    if count == len_featureGB:
        patient_lst.append(date_lst)
        seq_data.append(patient_lst)

print("output files")
# dump sequence features, patient ids, and labels to pickle files
if setType == "train":
    print("Dump train")
    pickle.dump(seq_data, open(os.path.join(PATH_TRAIN, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(patient_ids, open(os.path.join(PATH_TRAIN, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels, open(os.path.join(PATH_TRAIN, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
elif setType == "validation":
    print("Dump validation")
    pickle.dump(seq_data, open(os.path.join(PATH_VALID, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(patient_ids, open(os.path.join(PATH_VALID, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels, open(os.path.join(PATH_VALID, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
elif setType == "test":
    print("Dump test")
    pickle.dump(seq_data, open(os.path.join(PATH_TEST, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(patient_ids, open(os.path.join(PATH_TEST, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels, open(os.path.join(PATH_TEST, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
else:
    print("wrong dataset type")

print("Done")
