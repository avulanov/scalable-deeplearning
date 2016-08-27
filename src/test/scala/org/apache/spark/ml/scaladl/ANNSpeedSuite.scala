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

import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier => SMLP}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.scaladl.{MultilayerPerceptronClassifier => TMLP}
import org.scalatest.FunSuite

import scaladl.util.SparkTestContext

class ANNSpeedSuite extends FunSuite with SparkTestContext {

//  test ("speed test") {
//    val mnistPath = System.getenv("MNIST_HOME")
//    println(mnistPath + "/mnist.scale")
//    val dataFrame = sqlContext.
//      createDataFrame(MLUtils.loadLibSVMFile(sc, mnistPath + "/mnist.scale", 784)).persist()
//    dataFrame.count()
//    val mlp = new MultilayerPerceptronClassifier().setLayers(Array(784, 32, 10))
//      .setTol(10e-9)
//      .setMaxIter(20)
//      .setSeed(1234L)
//    val t = System.nanoTime()
//    val model = mlp.fit(dataFrame)
//    val total = System.nanoTime() - t
//    println("Total time: " + total / 1e9 + " s. (should be ~42s. without native BLAS")
//    val test = sqlContext.
//      createDataFrame(MLUtils.loadLibSVMFile(sc, mnistPath + "/mnist.scale.t", 784)).persist()
//    test.count()
//    val result = model.transform(test)
//    val pl = result.select("prediction", "label")
//    val ev = new MulticlassClassificationEvaluator().setMetricName("precision")
//    println("Accuracy: " + ev.evaluate(pl))
//  }

  test ("speed test with tensor (native BLAS and MNIST_HOME needs to be configured") {
    val mnistPath = System.getenv("MNIST_HOME")
    val dataFrame = spark
      .read
      .format("libsvm")
      .option("numFeatures", 784)
      .load(mnistPath + "/mnist.scale")
      .persist()
    dataFrame.count()
    val layers = Array(784, 100, 10)
    val maxIter = 20
    val tol = 1e-9
    val warmUp = new SMLP().setLayers(layers)
      .setTol(10e-9)
      .setMaxIter(1)
      .setSeed(1234L)
      .fit(dataFrame)
    val weights = warmUp.weights

    val mlp = new SMLP().setLayers(layers)
      .setTol(tol)
      .setMaxIter(maxIter)
      .setInitialWeights(weights.copy)
    val t = System.nanoTime()
    val model = mlp.fit(dataFrame)
    val total = System.nanoTime() - t
    val tensorMLP = new TMLP().setLayers(layers)
      .setTol(tol)
      .setMaxIter(maxIter)
      .setInitialWeights(weights.copy)
    val tTensor = System.nanoTime()
    val tModel = tensorMLP.fit(dataFrame)
    val totalTensor = System.nanoTime() - tTensor
    // time is 49.9 s on my machine
    assert(math.abs(totalTensor - total) / 1e9 < 0.15 * total /1e9,
      "Training time of tensor version should differ no more than 15% s. from original version")
    val test = spark
      .read
      .format("libsvm")
      .option("numFeatures", 784)
      .load(mnistPath + "/mnist.scale.t")
      .persist()
    test.count()
    val result = model.transform(test)
    val pl = result.select("prediction", "label")
    val ev = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val tResult = tModel.transform(test)
    val tpl = tResult.select("prediction", "label")
    val tev = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    assert(tev.evaluate(tpl) == ev.evaluate(pl), "Accuracies must be equal")
  }
}
