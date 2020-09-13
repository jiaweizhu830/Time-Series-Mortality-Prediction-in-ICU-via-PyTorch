/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.featureconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object FeatureConstructor {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
   * @return: Constructed Graph
   *
   */
  def constructFeatureMap(patients: RDD[Patient], labResults: RDD[LabResult],
    medications: RDD[Medication], diagnostics: RDD[Diagnostic],
    admissions: RDD[Admission], icustays: RDD[Icustay]): Map[String, Long] = {

    def checkArrayEmpty(baseIndex: Array[Long]): Long = {
      if (baseIndex.isEmpty)
        return 0
      else
        return baseIndex.max + 1
    }

    /**
     * Feature Map
     */

    //transform icd9 code
    def convert_icd9(str: String): String = {
      if (str.trim()(0) == 'E')
        return str.trim().slice(0, 4)
      else
        return str.trim().slice(0, 3)
    }

    //icd9 code features
    val icd9Array = diagnostics.
      map(diagnostic => diagnostic.icd9code).
      map(code => convert_icd9(code)).
      distinct.
      zipWithIndex.
      map(x => (x._1, x._2)).
      collect

    val medBaseIndex = icd9Array.size + 1
    val icd9Map = icd9Array.toMap

    //medication features
    val drugArray = medications.
      map(medication => medication.medicine.trim().toLowerCase()).
      distinct.
      zipWithIndex.
      map(x => (x._1, x._2 + medBaseIndex)).
      collect

    val labBaseIndex = drugArray.size + medBaseIndex
    val drugMap = drugArray.toMap

    //lab result features
    val labArray = labResults.
      map(lab => lab.labName.trim().toLowerCase()).
      distinct.
      zipWithIndex().
      map(x => (x._1, x._2 + labBaseIndex)).
      collect

    val ICUBaseIndex = labArray.size + labBaseIndex
    val labMap = labArray.toMap

    //length of ICU stay feature
    val ICUMap = Map("ICUStayDuration" -> ICUBaseIndex.toLong)

    //combine features
    val featureMap = icd9Map ++ drugMap ++ labMap ++ ICUMap

    featureMap
  }

  def createDataset(patients: RDD[Patient], labResults: RDD[LabResult],
    medications: RDD[Medication], diagnostics: RDD[Diagnostic],
    admissions: RDD[Admission], icustays: RDD[Icustay], featureMap: Map[String, Long], dataType: String) = {

    /**
     * Create Dataset
     */
    val spark = SparkSession.builder().
      getOrCreate()
    import spark.implicits._

    /**
     * input RDD format
     * patients, admission, diag, med, lab, icu
     * daignose_icd(subjectID, hadm_id, icd9_code)
     * prescription(subjectID, hadm_id, drug)
     * admissions(subjectID, hadm_Id, admittime)
     * labevents(subjectID, hadm_id, itemid, valuenum)
     * patients(subjectID, dod_hosp)
     * icustays(subjectID, hadm_id, icustay_id, LOS)
     */
    val patientRDD = patients.map(x => (x.patientID, x.dod_hosp))

    //transform icd9 code
    def convert_icd9(str: String): String = {
      if (str.trim()(0) == 'E')
        return str.trim().slice(0, 4)
      else
        return str.trim().slice(0, 3)
    }

    println("diag RDD in construct")
    //transform diagnostics rdd
    val diagRDD = diagnostics.map(x => ((x.patientID, x.hadmID), convert_icd9(x.icd9code)))
    // admRDD format: patientID, hadmID, admittime
    //transform admissions rdd
    val admRDD = admissions.map(x => ((x.patientID, x.hadmID), x.admittime))
    // diagTimeRDD format: patientID, admittime, icd9_code
    val diagTimeRDD = admRDD.join(diagRDD).map(x => (x._1._1, x._2._1, x._2._2, 1.0))

    println("med RDD in construct")
    //transform medications rdd
    val medRDD = medications.map(x => ((x.patientID, x.hadmID), x.medicine.trim().toLowerCase()))
    // medTimeRDD format: pateintID, admittime, drug
    val medTimeRDD = admRDD.join(medRDD).map(x => (x._1._1, x._2._1, x._2._2, 1.0))

    println("lab RDD in construct")
    //transform labResults rdd
    val labRDD = labResults.map(x => ((x.patientID, x.hadmID), (x.labName.trim().toLowerCase(), x.value)))
    // labTimeRDD format: patientID, admittime, (lab, value)
    val labTimeRDD = admRDD.join(labRDD).map(x => (x._1._1, x._2._1, x._2._2._1, x._2._2._2))

    println("icu RDD in construct")
    //transform icustays rdd
    val icuRDD = icustays.map(x => ((x.patientID, x.hadmID), x.LOS))
    // icuTimeRDD format: patientID, admittime, LOS
    val icuTimeRDD = admRDD.join(icuRDD).map(x => (x._1._1, x._2._1, "ICUStayDuration", x._2._2))

    println("union RDD")
    // allRDD format: patientID, adimittime, feature_name, value
    //union all rdds
    val allRDD = diagTimeRDD.union(medTimeRDD).
      union(labTimeRDD).union(icuTimeRDD)

    println("MAP RDD")
    // broadcast
    val sc = patients.sparkContext
    //broadcast feature map:  key -> feature name (String), value -> feature id (Long)
    val bc_featureMap = sc.broadcast(featureMap)
    val features = featureMap.map(feature => (feature._1)).toSet

    //map feature name to feature id, and sort allRDD by patientID and admittime
    val allRDDMapped = allRDD.filter(x => features(x._3)).
      map(x => (x._1, x._2, bc_featureMap.value(x._3), x._4)).
      sortBy(x => (x._1, x._2))

    println("patient RDD")
    //transform patients rdd
    val finalPatientSet = allRDDMapped.map(x => x._1).collect.toSet
    val finalPatientRDD = patientRDD.filter(x => finalPatientSet(x._1))

    println("before output to file in createdataset")
    //output rdd results to csv files
    if (dataType == "train") {
      println("train")
      println("convert feature CSV")
      allRDDMapped.toDF("patientID", "admitTime", "featureID", "featureValue").write.csv("featureDF_train")
      println("convert IDs label CSV")
      finalPatientRDD.toDF("patientID", "IS_DEAD").write.csv("patientIDLabels_train")
    } else if (dataType == "validation") {
      println("validation")
      println("convert feature CSV")
      allRDDMapped.toDF("patientID", "admitTime", "featureID", "featureValue").write.csv("featureDF_valid")
      println("convert IDs label CSV")
      finalPatientRDD.toDF("patientID", "IS_DEAD").write.csv("patientIDLabels_valid")
    } else if (dataType == "test") {
      println("test")
      println("convert feature CSV")
      allRDDMapped.toDF("patientID", "admitTime", "featureID", "featureValue").write.csv("featureDF_test")
      println("convert IDs label CSV")
      finalPatientRDD.toDF("patientID", "IS_DEAD").write.csv("patientIDLabels_test")
    } else {
      println("wrong dataset type")
    }
  }
}
