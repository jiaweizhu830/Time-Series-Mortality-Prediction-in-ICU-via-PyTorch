/**
 * @author Jiawei Zhu <jzhu360@gatech.edu>
 * reference: CSE6250 HW4
 */

package edu.gatech.cse6250.main

import java.text.SimpleDateFormat
import edu.gatech.cse6250.featureconstruct.FeatureConstructor
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import edu.gatech.cse6250.model._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext

    /** initialize loading of training data */
    println("initialize loading of training data")
    val loadTrain = "train"
    val (patientTrain, medicationTrain, labResultTrain, diagnosticTrain, admissionTrain, icustayTrain) = loadRddRawTrainData(spark, loadTrain)
    /** construct feature map*/
    println("construct feature map")
    val featureMap = FeatureConstructor.constructFeatureMap(patientTrain, labResultTrain, medicationTrain, diagnosticTrain, admissionTrain, icustayTrain)
    /** construct training data*/
    println("construct training data")
    FeatureConstructor.createDataset(patientTrain, labResultTrain, medicationTrain, diagnosticTrain, admissionTrain, icustayTrain, featureMap, loadTrain)

    /** construct validation data*/
    val loadValid = "validation"
    println("construct validation data")
    val (patientVali, medicationVali, labResultVali, diagnosticVali, admissionVali, icustayVali) = loadRddRawValidData(spark, loadValid)
    FeatureConstructor.createDataset(patientVali, labResultVali, medicationVali, diagnosticVali, admissionVali, icustayVali, featureMap, loadValid)

    /** construct test data*/
    val loadTest = "test"
    println("construct testing data")
    val (patientTest, medicationTest, labResultTest, diagnosticTest, admissionTest, icustayTest) = loadRddRawTestData(spark, loadTest)

    FeatureConstructor.createDataset(patientTest, labResultTest, medicationTest, diagnosticTest, admissionTest, icustayTest, featureMap, loadTest)

    sc.stop()
  }

  def loadRddRawTrainData(spark: SparkSession, loadType: String): (RDD[Patient], RDD[Medication], RDD[LabResult], RDD[Diagnostic], RDD[Admission], RDD[Icustay]) = {
    import spark.implicits._
    val sqlContext = spark.sqlContext

    val dateFormat = new SimpleDateFormat("yyyy-MM-dd")
    println("before loadtype condition")

    if (loadType == "train") {
      println("before load csv")
      List("train/patients_train.csv", "train/labevents_train.csv", "train/diagnoses_train.csv", "train/prescripts_train.csv", "train/admissions_train.csv", "train/icustays_train.csv")
        .foreach(CSVHelper.loadCSVAsTable(spark, _))
    }
    println("before SQL train")

    /** load training data*/
    val patient = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, IS_DEAD
        |FROM patients_train
        |WHERE SUBJECT_ID IS NOT NULL AND IS_DEAD IS NOT NULL
        |AND SUBJECT_ID <> '' AND IS_DEAD <> ''
      """.stripMargin)
      .map(r => Patient(r(0).toString.toLong, r(1).toString.toInt))

    val labResult = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ITEMID, VALUENUM
        |FROM labevents_train
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ITEMID IS NOT NULL AND VALUENUM IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ITEMID <> '' AND VALUENUM <> ''
      """.stripMargin)
      .map(r => LabResult(r(0).toString.toLong, r(1).toString, r(2).toString, r(3).toString.toDouble))

    val diagnostic = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ICD9_CODE
        |FROM diagnoses_train
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ICD9_CODE IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ICD9_CODE <> ''
      """.stripMargin)
      .map(r => Diagnostic(r(0).toString.toLong, r(1).toString, r(2).toString))

    val medication = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, DRUG
        |FROM prescripts_train
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND DRUG IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND DRUG <> ''
      """.stripMargin)
      .map(r => Medication(r(0).toString.toLong, r(1).toString, r(2).toString))

    val admission = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ADMITTIME
        |FROM admissions_train
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ADMITTIME IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ADMITTIME <> ''
      """.stripMargin)
      .map(r => Admission(r(0).toString.toLong, r(1).toString, dateFormat.parse(r(2).toString).getTime()))

    val icustay = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ICUSTAY_ID, LOS
        |FROM icustays_train
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ICUSTAY_ID IS NOT NULL AND LOS IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ICUSTAY_ID <> '' AND LOS <> ''
      """.stripMargin)
      .map(r => Icustay(r(0).toString.toLong, r(1).toString, r(2).toString, r(3).toString.toDouble))

    (patient.rdd, medication.rdd, labResult.rdd, diagnostic.rdd, admission.rdd, icustay.rdd)

  }

  def loadRddRawValidData(spark: SparkSession, loadType: String): (RDD[Patient], RDD[Medication], RDD[LabResult], RDD[Diagnostic], RDD[Admission], RDD[Icustay]) = {
    import spark.implicits._
    val sqlContext = spark.sqlContext

    val dateFormat = new SimpleDateFormat("yyyy-MM-dd")

    println("before loadtype condition")

    if (loadType == "validation") {
      List("valid/patients_valid.csv", "valid/labevents_valid.csv", "valid/diagnoses_valid.csv", "valid/prescripts_valid.csv", "valid/admissions_valid.csv", "valid/icustays_valid.csv")
        .foreach(CSVHelper.loadCSVAsTable(spark, _))
    }
    println("before SQL valid")

    /** load validation data*/
    val patient = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, IS_DEAD
        |FROM patients_valid
        |WHERE SUBJECT_ID IS NOT NULL AND IS_DEAD IS NOT NULL
        |AND SUBJECT_ID <> '' AND IS_DEAD <> ''
      """.stripMargin)
      .map(r => Patient(r(0).toString.toLong, r(1).toString.toInt))

    val labResult = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ITEMID, VALUENUM
        |FROM labevents_valid
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ITEMID IS NOT NULL AND VALUENUM IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ITEMID <> '' AND VALUENUM <> ''
      """.stripMargin)
      .map(r => LabResult(r(0).toString.toLong, r(1).toString, r(2).toString, r(3).toString.toDouble))

    val diagnostic = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ICD9_CODE
        |FROM diagnoses_valid
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ICD9_CODE IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ICD9_CODE <> ''
      """.stripMargin)
      .map(r => Diagnostic(r(0).toString.toLong, r(1).toString, r(2).toString))

    val medication = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, DRUG
        |FROM prescripts_valid
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND DRUG IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND DRUG <> ''
      """.stripMargin)
      .map(r => Medication(r(0).toString.toLong, r(1).toString, r(2).toString))

    val admission = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ADMITTIME
        |FROM admissions_valid
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ADMITTIME IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ADMITTIME <> ''
      """.stripMargin)
      .map(r => Admission(r(0).toString.toLong, r(1).toString, dateFormat.parse(r(2).toString).getTime()))

    val icustay = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ICUSTAY_ID, LOS
        |FROM icustays_valid
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ICUSTAY_ID IS NOT NULL AND LOS IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ICUSTAY_ID <> '' AND LOS <> ''
      """.stripMargin)
      .map(r => Icustay(r(0).toString.toLong, r(1).toString, r(2).toString, r(3).toString.toDouble))

    (patient.rdd, medication.rdd, labResult.rdd, diagnostic.rdd, admission.rdd, icustay.rdd)

  }

  def loadRddRawTestData(spark: SparkSession, loadType: String): (RDD[Patient], RDD[Medication], RDD[LabResult], RDD[Diagnostic], RDD[Admission], RDD[Icustay]) = {
    import spark.implicits._
    val sqlContext = spark.sqlContext

    val dateFormat = new SimpleDateFormat("yyyy-MM-dd")

    /** load testing data*/

    println("before loadtype condition")

    if (loadType == "test") {
      List("test/patients_test.csv", "test/labevents_test.csv", "test/diagnoses_test.csv", "test/prescripts_test.csv", "test/admissions_test.csv", "test/icustays_test.csv")
        .foreach(CSVHelper.loadCSVAsTable(spark, _))
    }
    println("before SQL test")

    val patient = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, IS_DEAD
        |FROM patients_test
        |WHERE SUBJECT_ID IS NOT NULL AND IS_DEAD IS NOT NULL
        |AND SUBJECT_ID <> '' AND IS_DEAD <> ''
      """.stripMargin)
      .map(r => Patient(r(0).toString.toLong, r(1).toString.toInt))

    val labResult = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ITEMID, VALUENUM
        |FROM labevents_test
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ITEMID IS NOT NULL AND VALUENUM IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ITEMID <> '' AND VALUENUM <> ''
      """.stripMargin)
      .map(r => LabResult(r(0).toString.toLong, r(1).toString, r(2).toString, r(3).toString.toDouble))

    val diagnostic = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ICD9_CODE
        |FROM diagnoses_test
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ICD9_CODE IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ICD9_CODE <> ''
      """.stripMargin)
      .map(r => Diagnostic(r(0).toString.toLong, r(1).toString, r(2).toString))

    val medication = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, DRUG
        |FROM prescripts_test
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND DRUG IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND DRUG <> ''
      """.stripMargin)
      .map(r => Medication(r(0).toString.toLong, r(1).toString, r(2).toString))

    val admission = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ADMITTIME
        |FROM admissions_test
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ADMITTIME IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ADMITTIME <> ''
      """.stripMargin)
      .map(r => Admission(r(0).toString.toLong, r(1).toString, dateFormat.parse(r(2).toString).getTime()))

    val icustay = sqlContext.sql(
      """
        |SELECT SUBJECT_ID, HADM_ID, ICUSTAY_ID, LOS
        |FROM icustays_test
        |WHERE SUBJECT_ID IS NOT NULL AND HADM_ID IS NOT NULL AND ICUSTAY_ID IS NOT NULL AND LOS IS NOT NULL
        |AND SUBJECT_ID <> '' AND HADM_ID <> '' AND ICUSTAY_ID <> '' AND LOS <> ''
      """.stripMargin)
      .map(r => Icustay(r(0).toString.toLong, r(1).toString, r(2).toString, r(3).toString.toDouble))

    (patient.rdd, medication.rdd, labResult.rdd, diagnostic.rdd, admission.rdd, icustay.rdd)

  }

}
