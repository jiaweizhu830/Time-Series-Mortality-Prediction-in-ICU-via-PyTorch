import os
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *


def process(input_path, output_path="data/", split_weight=[65.0, 20.0, 15.0], seed=1):
    data_files = ["ADMISSIONS.csv", "DIAGNOSES_ICD.csv", "ICUSTAYS.csv", 
                  "LABEVENTS.csv", "PATIENTS.csv", "PRESCRIPTIONS.csv"]

    spark = SparkSession.builder.appName("BD4H_Project").getOrCreate()

    # Load dataset with specific columns
    admissions = spark.read.csv(input_path + data_files[0], inferSchema=False, header=True) \
                      .select(["SUBJECT_ID", "HADM_ID", "ADMITTIME"])
    diagnoses  = spark.read.csv(input_path + data_files[1], inferSchema=False, header=True) \
                      .select(["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
    icustays   = spark.read.csv(input_path + data_files[2], inferSchema=False, header=True) \
                      .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "LOS"])
    labevents  = spark.read.csv(input_path + data_files[3], inferSchema=False, header=True) \
                      .select(["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM"])
    patients   = spark.read.csv(input_path + data_files[4], inferSchema=False, header=True) \
                      .select(["SUBJECT_ID", "DOD_HOSP"])
    prescripts = spark.read.csv(input_path + data_files[5], inferSchema=False, header=True) \
                      .select(["SUBJECT_ID", "HADM_ID", "DRUG"])

    # Process columns as required and deal with null values
    icustays   = icustays.withColumn("LOS", icustays["LOS"].cast(DoubleType())) \
                         .fillna(0, subset=["LOS"])
    labevents  = labevents.withColumn("VALUENUM", labevents["VALUENUM"].cast(DoubleType())) \
                          .filter(labevents["VALUENUM"].isNotNull())
    patients   = patients.withColumn("IS_DEAD", when(patients["DOD_HOSP"].isNull(), 0).otherwise(1))

    # Compute subject_id for ICU patients
    icu_patient_ids = icustays.filter(icustays["ICUSTAY_ID"].isNotNull()) \
                              .select(["SUBJECT_ID"]).distinct()

    # Split dataset base on subject_id, train:test:validation = split_weight
    id_splits = icu_patient_ids.randomSplit(split_weight, seed=seed)

    # Compute training, testing and validation set
    data_splits = {}
    prefix = ["admissions", "diagnoses", "icustays", "labevents", "patients", "prescripts"]

    for i, category in enumerate(["train", "test", "valid"]):
        data_splits[prefix[0] + "_" + category] = admissions.join(id_splits[i], 
            on="SUBJECT_ID", how="inner").select(["SUBJECT_ID", "HADM_ID", "ADMITTIME"])

        data_splits[prefix[1] + "_" + category] = diagnoses.join(id_splits[i], 
            on="SUBJECT_ID", how="inner").select(["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])

        data_splits[prefix[2] + "_" + category] = icustays.join(id_splits[i], 
            on="SUBJECT_ID", how="inner").select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "LOS"])

        labevents_tmp = labevents.join(id_splits[i], on="SUBJECT_ID", how="inner") \
                                 .select(["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM"])
        
        # Normalize valuenum in labevents
        max_valuenum = labevents_tmp.agg(max(labevents_tmp["VALUENUM"])).first()[0]
        data_splits[prefix[3] + "_" + category] = labevents_tmp.withColumn("VALUENUM", labevents_tmp["VALUENUM"] / max_valuenum)

        data_splits[prefix[4] + "_" + category] = patients.join(id_splits[i], 
            on="SUBJECT_ID", how="inner").select(["SUBJECT_ID", "IS_DEAD"])

        data_splits[prefix[5] + "_" + category] = prescripts.join(id_splits[i], 
            on="SUBJECT_ID", how="inner").select(["SUBJECT_ID", "HADM_ID", "DRUG"])

    # Save splited dataset to the train, test and valid subfolders in the output path
    for i, category in enumerate(["train", "test", "valid"]):
        os.makedirs(os.path.join(output_path, category), exist_ok=True)

    for i, category in enumerate(["train", "test", "valid"]):
        for name in prefix:
            output_filename = name + "_" + category + ".csv"
            to_path = os.path.join(output_path, category, name)

            # Output the dataframe to a single csv file
            data_splits[name + "_" + category].repartition(1).write.csv(to_path, header=True)

            # Rename the output file
            for filename in os.listdir(to_path):
                if (filename.startswith("part") and filename.endswith("csv")):
                    os.rename(os.path.join(to_path, filename), os.path.join(to_path, output_filename))
                    break


if __name__ == "__main__":
    input_path = "data/"
    output_path = "data/"
    split_weight = [65.0, 20.0, 15.0] # Must be list of doubles
    seed = 1

    process(input_path, output_path, split_weight, seed)