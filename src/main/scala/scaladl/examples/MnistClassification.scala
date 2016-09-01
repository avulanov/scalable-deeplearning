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
import org.apache.spark.ml.scaladl.MultilayerPerceptronClassifier
import org.apache.spark.sql.SparkSession

object MnistClassification {

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
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    // train the model
    val model = trainer.fit(train)
    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    // scalastyle:off
    println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
    // scalastyle:on
  }
}
