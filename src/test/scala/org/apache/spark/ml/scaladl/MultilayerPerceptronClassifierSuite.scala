/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.scaladl

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.scalatest.FunSuite

import scaladl.util.SparkTestContext

class MultilayerPerceptronClassifierSuite extends FunSuite with SparkTestContext {

  test("XOR function learning as binary classification problem with two outputs.") {
    val dataFrame = spark.createDataFrame(Seq(
      (Vectors.dense(0.0, 0.0), 0.0),
      (Vectors.dense(0.0, 1.0), 1.0),
      (Vectors.dense(1.0, 0.0), 1.0),
      (Vectors.dense(1.0, 1.0), 0.0))
    ).toDF("features", "label")
    val layers = Array[Int](2, 5, 2)
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(1)
      .setSeed(123L)
      .setMaxIter(100)
    val model = trainer.fit(dataFrame)
    val result = model.transform(dataFrame)
    val predictionAndLabels = result.select("prediction", "label").collect()
    predictionAndLabels.foreach { case Row(p: Double, l: Double) =>
      assert(p == l)
    }
  }

  test("Test setWeights by training restart") {
    val dataFrame = spark.createDataFrame(Seq(
      (Vectors.dense(0.0, 0.0), 0.0),
      (Vectors.dense(0.0, 1.0), 1.0),
      (Vectors.dense(1.0, 0.0), 1.0),
      (Vectors.dense(1.0, 1.0), 0.0))
    ).toDF("features", "label")
    val layers = Array[Int](2, 5, 2)
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(1)
      .setSeed(123456L)
      .setMaxIter(1)
      .setTol(1e-6)
    val initialWeights = trainer.fit(dataFrame).weights
    trainer.setInitialWeights(initialWeights.copy)
    val weights1 = trainer.fit(dataFrame).weights
    trainer.setInitialWeights(initialWeights.copy)
    val weights2 = trainer.fit(dataFrame).weights
    weights1.toArray.zip(weights2.toArray).foreach { x =>
      assert(math.abs(x._1 - x._2) <= 10e-5,
        "Training should produce the same weights given equal initial weights and number of steps")
    }
  }
}
