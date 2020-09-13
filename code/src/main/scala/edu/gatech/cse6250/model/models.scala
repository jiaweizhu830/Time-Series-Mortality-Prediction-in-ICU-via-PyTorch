/**
 * Reference: CSE6250 HW4
 */

package edu.gatech.cse6250.model

case class LabResult(patientID: Long, hadmID: String, labName: String, value: Double)

case class Diagnostic(patientID: Long, hadmID: String, icd9code: String)

case class Medication(patientID: Long, hadmID: String, medicine: String)

case class Admission(patientID: Long, hadmID: String, admittime: Long)

case class Icustay(patientID: Long, hadmID: String, icustayID: String, LOS: Double)

case class Patient(patientID: Long, dod_hosp: Int)
