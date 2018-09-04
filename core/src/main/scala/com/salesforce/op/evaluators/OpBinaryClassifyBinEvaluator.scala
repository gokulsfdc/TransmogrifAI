/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.salesforce.op.evaluators

import com.salesforce.op.UID
import org.apache.spark.ml.linalg.Vector

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.DoubleType
import org.slf4j.LoggerFactory

/**
  *
  * Instance to evaluate OpBinaryClassifyBin metrics
  * The metrics are BrierScore, AverageScore, count per bin
  * Default evaluation returns sum of each bin's BrierScore
  *
  * @param name name of default metric
  * @param isLargerBetter is metric better if larger
  * @param uid uid for instance
  */
private[op] class OpBinaryClassifyBinEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.Binary,
  override val isLargerBetter: Boolean = true,
  override val uid: String = UID[OpBinaryClassificationEvaluator],
  val numBins: Int = 100
) extends OpBinaryClassificationEvaluatorBase[Seq[OpBinStatsEvaluator]](uid = uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: Seq[OpBinStatsEvaluator] => Double = _.map(_.BrierScore).sum

  override def evaluateAll(data: Dataset[_]): Seq[OpBinStatsEvaluator] = {
    val labelColName = getLabelCol
    val dataUse = makeDataToUse(data, labelColName)

    val (rawPredictionColName, predictionColName, probabilityColName) =
      (getRawPredictionCol, getPredictionValueCol, getProbabilityCol)
    log.debug(
      "Evaluating metrics on columns :\n label : {}\n rawPrediction : {}\n prediction : {}\n probability : {}\n",
      labelColName, rawPredictionColName, predictionColName, probabilityColName
    )

    import dataUse.sparkSession.implicits._
    val rdd = dataUse.select(predictionColName, labelColName).as[(Double, Double)].rdd

    if (rdd.isEmpty()) {
      log.error("The dataset is empty")
      Seq.empty[OpBinStatsEvaluator]
    } else {
      val scoreAndLabels =
        dataUse.select(col(probabilityColName), col(labelColName).cast(DoubleType)).rdd.map {
          case Row(prob: Vector, label: Double) => (prob(1), label)
          case Row(prob: Double, label: Double) => (prob, label)
        }

      // TODO: Round off the scores.

      // Compute squared error for each data point. Data is change to (score, (score, squaredError)).
      val dataWithSquaredError = for {i <- scoreAndLabels} yield (i._1, (i._1, math.pow(i._1 - i._2, 2)))

      // Combine data based on score. For each score value,
      // sum up its squared error, sum up its score and count the number of data points for a given score.
      // Data is represented as (Score, (SumOfScores, SumOfSquaredErrors, NumberOfDataPoints))
      val counts = dataWithSquaredError.combineByKey(
        (s: (Double, Double)) => (s._1, s._2, 1),
        (s1: (Double, Double, Int), s2: (Double, Double)) => (s1._1 + s2._1, s1._2 + s2._2, s1._3 + 1),
        (s1: (Double, Double, Int), s2: (Double, Double, Int)) => (s1._1 + s2._1, s1._2 + s2._2, s1._3 + s2._3)
      ).sortByKey(ascending = false)

      // Check if partitioning the data points is required.
      val countsSize = counts.count()
      val isPartitionRequired: Boolean = isPartitionRequired(countsSize)
      if (isPartitionRequired == false) {
        val metrics: Seq[OpBinStatsEvaluator] = counts.map { scoreDetails =>
          OpBinStatsEvaluator(
            NumberOfDataPoints = scoreDetails._2._3,
            AvgScore = scoreDetails._2._1 / scoreDetails._2._3,
            BrierScore = scoreDetails._2._2 / scoreDetails._2._3
          )
        }.collect()
        metrics
      }

      var grouping = countsSize / numBins
      if (grouping >= Int.MaxValue) {
        log.info("Curve is too large {} for {} bins. capping at {}", countsSize, numBins, Int.MaxValue)
        grouping = Int.MaxValue
      }

      // Partition the data into individual bins and for each bin count the number of data points,
      // average score predicted and the brier score of the bin values.
      val metrics: Seq[OpBinStatsEvaluator] = counts.mapPartitions(_.grouped(grouping.toInt).map { scoreDetails =>
        val (totalScore, sumOfSquaredError, numberOfDataPoints) = scoreDetails.foldLeft(0.0, 0.0, 0)(
          (r: (Double, Double, Int), s: (Double, (Double, Double, Int))) => (r._1 + s._2._1, r._2 + s._2._2, r._3 + s._2._3)
        )

        OpBinStatsEvaluator(
          NumberOfDataPoints = numberOfDataPoints,
          AvgScore = totalScore / numberOfDataPoints,
          BrierScore = sumOfSquaredError / numberOfDataPoints
        )
      }).collect()

      log.info("Evaluated metrics: {}", metrics.toString)
      metrics
    }

  }

  // Check whether to partition the data into bins or not
  private def isPartitionRequired(countsSize: Int) : Boolean = {
    if (numBins == 0) {
      false
    }

    var grouping = countsSize / numBins
    if ( grouping < 2) {
      log.info("Number of Groups: {}. Curve is too small {} for {} bins", grouping, countsSize, numBins)
      false
    }

    true
  }
}

/**
  * Metrics of OpBinStatsEvaluator
  *
  * @param NumberOfDataPoints
  * @param AvgScore
  * @param BrierScore
  */
case class OpBinStatsEvaluator
(
  NumberOfDataPoints: Int,
  AvgScore: Double,
  BrierScore: Double
) extends EvaluationMetrics
