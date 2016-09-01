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

package scaladl.examples

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.scaladl.{MultilayerPerceptronClassifier, StackedAutoencoder}
import org.apache.spark.sql.SparkSession

object MnistEncoding {

  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      System.exit(0)
    }
    val mnistPath = args(0)
    val spark = SparkSession.builder
      .appName("my-spark-app")
      .config("spark.sql.warehouse.dir", "warehouse-temp")
      .getOrCreate()
    val mnistTrain = mnistPath + "/mnist.scale"
    val mnistTest = mnistPath + "/mnist.scale.t"
    // Load the data stored in LIBSVM format as a DataFrame.
    // MNIST handwritten recognition data
    // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    val train = spark.read.format("libsvm").option("numFeatures", 784).load(mnistTrain).persist()
    val test = spark.read.format("libsvm").option("numFeatures", 784).load(mnistTest).persist()
    // materialize data lazily persisted in memory
    train.count()
    test.count()
    // specify layers for the neural network:
    // input layer of size 784 (features), one hidden layer of size 100
    // and output of size 10 (classes)
    val layers = Array[Int](784, 32, 10)
    // create autoencoder and decode with one hidden layer of 32 neurons
    val stackedAutoencoder = new StackedAutoencoder()
      .setLayers(layers.init)
      .setBlockSize(128)
      .setMaxIter(1)
      .setSeed(333L)
      .setTol(1e-6)
      .setInputCol("features")
      .setOutputCol("output")
      .setDataIn01Interval(true)
      .setBuildDecoder(false)
    val saModel = stackedAutoencoder.fit(train)
    val autoWeights = saModel.encoderWeights
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(123456789L)
      .setMaxIter(1)
      .setTol(1e-6)
    val initialWeights = trainer.fit(train).weights
    System.arraycopy(
      autoWeights.toArray, 0, initialWeights.toArray, 0, autoWeights.toArray.length)
    trainer
      .setInitialWeights(initialWeights)
      .setMaxIter(10)
      .setTol(1e-6)
    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    // scalastyle:off
    println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
    // scalastyle:on
  }
}
