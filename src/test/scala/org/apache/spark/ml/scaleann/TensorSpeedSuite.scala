package org.apache.spark.ml.scaleann

import org.apache.spark.ml.anntensor.{MultilayerPerceptronClassifier => TMLP}
import org.apache.spark.ml.util.SparkTestContext
import org.scalatest.FunSuite

class TensorSpeedSuite  extends FunSuite with SparkTestContext {

  test("tensor speed") {
    val mnistPath = System.getenv("MNIST_HOME")
    val dataFrame = spark
      .read
      .format("libsvm")
      .option("numFeatures", 784)
      .load(mnistPath + "/mnist.scale")
      .persist()
    dataFrame.count()
    val warmUp = new TMLP().setLayers(Array(784, 32, 10))
      .setTol(10e-9)
      .setMaxIter(2)
      .setSeed(1234L)
      .fit(dataFrame)
  }

}
