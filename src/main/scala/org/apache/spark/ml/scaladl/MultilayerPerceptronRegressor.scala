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

import scala.collection.JavaConverters._

import org.apache.hadoop.fs.Path

import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.{max, min}
import org.apache.spark.sql.types._

import scaladl.layers.{FeedForwardTopology, FeedForwardTrainer}

 /**
  * Params that need to mixin with both MultilayerPerceptronRegressorModel and
  * MultilayerPerceptronRegressor
  */
private[ml] trait MultilayerPerceptronRegressorParams extends PredictorParams {

 /**
  * Param indicating whether to scale the labels to be between 0 and 1.
  *
  * @group param
  */
 final val stdLabels: BooleanParam = new BooleanParam(
   this, "stdLabels", "Whether to standardize the dataset's labels to between 0 and 1.")

 /** @group getParam */
 def setStandardizeLabels(value: Boolean): this.type = set(stdLabels, value)

 /** @group getParam */
 def getStandardizeLabels: Boolean = $(stdLabels)

  setDefault(stdLabels -> true)
}

/** Label to vector converter. */
private object RegressionLabelConverter {

  var minimum = 0.0
  var maximum = 0.0
  /**
   * Encodes a label as a vector.
   * Returns a vector of length 1 with the label in the 0th position
   *
   * @param labeledPoint labeled point
   * @return pair of features and vector encoding of a label
   */
  def encodeLabeledPoint(labeledPoint: LabeledPoint, min: Double, max: Double,
         model: MultilayerPerceptronRegressor): (Vector, Vector) = {
    val output = Array.fill(1)(0.0)
    if (model.getStandardizeLabels) {
      minimum = min
      maximum = max
      output(0) = (labeledPoint.label - min) / (max - min)
    }
    else {
    // When min and max are equal, cannot min-max scale due to divide by zero error. Setting scaled
    // result to zero will lead to consistent predictions, as the min will be added during decoding.
    // Min and max will both be 0 if label scaling is turned off, and this code branch will run.
      output(0) = labeledPoint.label
    }
    (labeledPoint.features, Vectors.dense(output))
  }

  /**
   * Converts a vector to a label.
   * Returns the value of the 0th element of the output vector.
   *
   * @param output label encoded with a vector
   * @return label
   */
  def decodeLabel(output: Vector, model: MultilayerPerceptronRegressorModel): Double = {
    if (model.getStandardizeLabels) {
      (output(0) * (maximum - minimum)) + minimum
    } else {
      output(0)
    }
  }
}

 /**
  * :: Experimental ::
  * Regression trainer based on Multi-layer perceptron regression.
  * Contains sigmoid activation function on all layers, output layer has a linear function.
  * Number of inputs has to be equal to the size of feature vectors.
  * Number of outputs has to be equal to one.
  */
class MultilayerPerceptronRegressor (
    override val uid: String)
  extends Predictor[Vector, MultilayerPerceptronRegressor, MultilayerPerceptronRegressorModel]
    with MultilayerPerceptronParams with MultilayerPerceptronRegressorParams with Serializable
    with DefaultParamsWritable {

   /** @group setParam */
   def setLayers(value: Array[Int]): this.type = set(layers, value)

   /** @group setParam */
   def setBlockSize(value: Int): this.type = set(blockSize, value)

   /**
    * Set the maximum number of iterations.
    * Default is 100.
    *
    * @group setParam
    */
   def setMaxIter(value: Int): this.type = set(maxIter, value)

   /**
    * Set the convergence tolerance of iterations.
    * Smaller value will lead to higher accuracy with the cost of more iterations.
    * Default is 1E-4.
    *
    * @group setParam
    */
   def setTol(value: Double): this.type = set(tol, value)

   /**
    * Set the seed for weights initialization if weights are not set
    *
    * @group setParam
    */
   def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Sets the value of param [[initialWeights]].
   *
   * @group expertSetParam
   */
  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)

  /**
   * Sets the value of param [[optimizer]].
   * Default is "LBFGS".
   *
   * @group expertSetParam
   */
  def setOptimizer(value: String): this.type = set(optimizer, value)

  /**
   * Sets the value of param [[learningRate]] (applicable only for solver "gd").
   * Default is 0.03.
   *
   * @group setParam
   */
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def this() = this(Identifiable.randomUID("mlpr"))

  override def copy(extra: ParamMap): MultilayerPerceptronRegressor = defaultCopy(extra)

  /**
   * Train a model using the given dataset and parameters.
   *
   * @param dataset Training dataset
   * @return Fitted model
   */
  override protected def train(dataset: Dataset[_]): MultilayerPerceptronRegressorModel = {
    val myLayers = getLayers
    val lpData: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    // Compute minimum and maximum values in the training labels for scaling.
    val data = {
      if (getStandardizeLabels) {
        val minmax = dataset
          .agg(max("label").cast(DoubleType), min("label").cast(DoubleType)).collect()(0)
        // Encode and scale labels to prepare for training.
        lpData.map(lp =>
          RegressionLabelConverter.encodeLabeledPoint(lp, minmax(1).asInstanceOf[Double],
            minmax(0).asInstanceOf[Double], this))
      } else {
        lpData.map(lp =>
          RegressionLabelConverter.encodeLabeledPoint(lp, 0.0, 0.0, this))
      }
    }
    // Initialize the network architecture with the specified layer count and sizes.
    val topology = FeedForwardTopology.multiLayerPerceptronRegression(myLayers)
    // Prepare the Network trainer based on our settings.
    val trainer = new FeedForwardTrainer(topology, myLayers(0), myLayers.last)
    if (isDefined(initialWeights)) {
      trainer.setWeights($(initialWeights))
    } else {
      trainer.setSeed($(seed))
    }
     if (getOptimizer == "LBFGS") {
       trainer.LBFGSOptimizer
         .setConvergenceTol($(tol))
         .setNumIterations($(maxIter))
     } else if (getOptimizer == "GD") {
       trainer.SGDOptimizer
         .setNumIterations($(maxIter))
         .setConvergenceTol($(tol))
         .setStepSize($(learningRate))
     } else {
       throw new IllegalArgumentException(
         s"The solver $optimizer is not supported by MultilayerPerceptronRegressor.")
     }
    trainer.setStackSize($(blockSize))
    // Train Model.
    val mlpModel = trainer.train(data)
    new MultilayerPerceptronRegressorModel(uid, myLayers, mlpModel.weights)
  }
}


object MultilayerPerceptronRegressor
  extends DefaultParamsReadable[MultilayerPerceptronRegressor] {

  /** String name for "l-bfgs" solver. */
  private[ml] val LBFGS = "l-bfgs"

  /** String name for "gd" (minibatch gradient descent) solver. */
  private[ml] val GD = "gd"

  /** Set of solvers that MultilayerPerceptronRegressor supports. */
  private[ml] val supportedSolvers = Array(LBFGS, GD)

  override def load(path: String): MultilayerPerceptronRegressor = super.load(path)
}


/**
 * :: Experimental ::
 * Multi-layer perceptron regression model.
 * Each layer has sigmoid activation function, output layer has softmax.
 *
 * @param uid uid
 * @param layers array of layer sizes including input and output
 * @param weights weights (or parameters) of the model
 * @return prediction model
 */
class MultilayerPerceptronRegressorModel private[ml] (
    override val uid: String,
    val layers: Array[Int],
    val weights: Vector)
  extends PredictionModel[Vector, MultilayerPerceptronRegressorModel]
    with Serializable with MultilayerPerceptronRegressorParams with MLWritable {

  override val numFeatures: Int = layers.head

  private val mlpModel =
    FeedForwardTopology.multiLayerPerceptronRegression(layers).model(weights)

  /** Returns layers in a Java List. */
  private[ml] def javaLayers: java.util.List[Int] = layers.toList.asJava

  /**
   * Predict label for the given features.
   * This internal method is used to implement [[transform()]] and output [[predictionCol]].
   */
  override def predict(features: Vector): Double = {
    RegressionLabelConverter.decodeLabel(mlpModel.predict(features), this)
  }

  override def copy(extra: ParamMap): MultilayerPerceptronRegressorModel = {
    copyValues(new MultilayerPerceptronRegressorModel(uid, layers, weights), extra)
  }

  override def write: MLWriter =
  new MultilayerPerceptronRegressorModel.MultilayerPerceptronRegressorModelWriter(this)
}

object MultilayerPerceptronRegressorModel
  extends MLReadable[MultilayerPerceptronRegressorModel] {

  override def read: MLReader[MultilayerPerceptronRegressorModel] =
    new MultilayerPerceptronRegressorModelReader

  override def load(path: String): MultilayerPerceptronRegressorModel = super.load(path)

  /** [[MLWriter]] instance for [[MultilayerPerceptronRegressorModel]] */
  private[MultilayerPerceptronRegressorModel]
  class MultilayerPerceptronRegressorModelWriter(
    instance: MultilayerPerceptronRegressorModel) extends MLWriter {

    private case class Data(layers: Array[Int], weights: Vector)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: layers, weights
      val data = Data(instance.layers, instance.weights)
      val dataPath = new Path(path, "data").toString
      sqlContext.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class MultilayerPerceptronRegressorModelReader
    extends MLReader[MultilayerPerceptronRegressorModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[MultilayerPerceptronRegressorModel].getName

    override def load(path: String): MultilayerPerceptronRegressorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sqlContext.read.parquet(dataPath).select("layers", "weights").head()
      val layers = data.getAs[Seq[Int]](0).toArray
      val weights = data.getAs[Vector](1)
      val model = new MultilayerPerceptronRegressorModel(metadata.uid, layers, weights)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
