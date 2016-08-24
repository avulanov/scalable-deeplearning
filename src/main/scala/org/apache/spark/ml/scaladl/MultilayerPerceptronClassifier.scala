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

import scaladl.layers.{FeedForwardTopology, FeedForwardTrainer}

import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntArrayParam, IntParam, Param, ParamMap,
ParamValidators}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasSeed, HasTol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

/**
 * Params for Multilayer Perceptron.
 */
private[ml] trait MultilayerPerceptronParams extends PredictorParams
  with HasSeed with HasMaxIter with HasTol {
  /**
   * Layer sizes including input size and output size.
   * Default: Array(1, 1)
   *
   * @group param
   */
  final val layers: IntArrayParam = new IntArrayParam(this, "layers",
    "Sizes of layers from input layer to output layer" +
      " E.g., Array(780, 100, 10) means 780 inputs, " +
      "one hidden layer with 100 neurons and output layer of 10 neurons.",
    // TODO: how to check ALSO that all elements are greater than 0?
    ParamValidators.arrayLengthGt(1)
  )

  /** @group getParam */
  final def getLayers: Array[Int] = $(layers)

  /**
   * Block size for stacking input data in matrices to speed up the computation.
   * Data is stacked within partitions. If block size is more than remaining data in
   * a partition then it is adjusted to the size of this data.
   * Recommended size is between 10 and 1000.
   * Default: 128
   *
   * @group expertParam
   */
  final val blockSize: IntParam = new IntParam(this, "blockSize",
    "Block size for stacking input data in matrices. Data is stacked within partitions." +
      " If block size is more than remaining data in a partition then " +
      "it is adjusted to the size of this data. Recommended size is between 10 and 1000",
    ParamValidators.gt(0))

  /** @group getParam */
  final def getBlockSize: Int = $(blockSize)

  /**
   * Optimizer setup.
   *
   * @group expertParam
   */
  final val optimizer: Param[String] = new Param[String](this, "optimizer",
    " Allows setting the optimizer: minibatch gradient descent (GD) or LBFGS. " +
      " The latter is recommended one. ",
    ParamValidators.inArray[String](Array("GD", "LBFGS")))

  /** @group getParam */
  final def getOptimizer: String = $(optimizer)

  /**
   * Learning rate.
   *
   * @group expertParam
   */
  final val learningRate: DoubleParam = new DoubleParam(this, "learning rate",
    " Sets the learning rate for gradient descent optimizer ",
    ParamValidators.inRange(0, 1))

  /** @group getParam */
  final def getLearningRate: Double = $(learningRate)


  /**
   * The initial weights of the model.
   *
   * @group expertParam
   */
  final val initialWeights: Param[Vector] = new Param[Vector](this, "initialWeights",
    "The initial weights of the model")

  /** @group expertGetParam */
  final def getInitialWeights: Vector = $(initialWeights)

  setDefault(maxIter -> 100, tol -> 1e-6, layers -> Array(1, 1), blockSize -> 128,
    optimizer -> "LBFGS", learningRate -> 0.03)
}

/** Label to vector converter. */
private object LabelConverter {
  // TODO: Use OneHotEncoder instead
  /**
   * Encodes a label as a vector.
   * Returns a vector of given length with zeroes at all positions
   * and value 1.0 at the position that corresponds to the label.
   *
   * @param labeledPoint labeled point
   * @param labelCount total number of labels
   * @return pair of features and vector encoding of a label
   */
  def encodeLabeledPoint(labeledPoint: LabeledPoint, labelCount: Int): (Vector, Vector) = {
    val output = Array.fill(labelCount)(0.0)
    output(labeledPoint.label.toInt) = 1.0
    (labeledPoint.features, Vectors.dense(output))
  }

  /**
   * Converts a vector to a label.
   * Returns the position of the maximal element of a vector.
   *
   * @param output label encoded with a vector
   * @return label
   */
  def decodeLabel(output: Vector): Double = {
    output.argmax.toDouble
  }
}

/**
 * Classifier trainer based on the Multilayer Perceptron.
 * Each layer has sigmoid activation function, output layer has softmax.
 * Number of inputs has to be equal to the size of feature vectors.
 * Number of outputs has to be equal to the total number of labels.
 *
 */
class MultilayerPerceptronClassifier (override val uid: String)
  extends Predictor[Vector, MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel]
    with MultilayerPerceptronParams {

  def this() = this(Identifiable.randomUID("mlpc-scaladl"))

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
   * Generate weights.
   */
  def generateWeights(): Vector = {
    val topology = FeedForwardTopology.multiLayerPerceptron($(layers), true)
    topology.model($(seed)).weights
  }

  override def copy(extra: ParamMap): MultilayerPerceptronClassifier = defaultCopy(extra)

  /**
   * Train a model using the given dataset and parameters.
   * Developers can implement this instead of [[fit()]] to avoid dealing with schema validation
   * and copying parameters into the model.
   *
   * @param dataset Training dataset
   * @return Fitted model
   */
  override protected def train(dataset: Dataset[_]): MultilayerPerceptronClassificationModel = {
    val myLayers = $(layers)
    val labels = myLayers.last
    val lpData = extractLabeledPoints(dataset)
    val data = lpData.map(lp => LabelConverter.encodeLabeledPoint(lp, labels))
    val topology = FeedForwardTopology.multiLayerPerceptron(myLayers, true)
    val trainer = new FeedForwardTrainer(topology, myLayers(0), myLayers.last)
    if (isDefined(initialWeights)) {
      trainer.setWeights($(initialWeights))
    } else {
      trainer.setSeed($(seed))
    }
    trainer.LBFGSOptimizer
      .setConvergenceTol($(tol))
      .setNumIterations($(maxIter))
    trainer.setStackSize($(blockSize))
    val mlpModel = trainer.train(data)
    new MultilayerPerceptronClassificationModel(uid, myLayers, mlpModel.weights)
  }
}

/**
 * Classification model based on the Multilayer Perceptron.
 * Each layer has sigmoid activation function, output layer has softmax.
 *
 * @param uid uid
 * @param layers array of layer sizes including input and output layers
 * @param weights vector of initial weights for the model that consists of the weights of layers
 * @return prediction model
 */
class MultilayerPerceptronClassificationModel private[ml] (override val uid: String,
                                                           val layers: Array[Int],
                                                           val weights: Vector)
  extends PredictionModel[Vector, MultilayerPerceptronClassificationModel]
    with Serializable {

  override val numFeatures: Int = layers.head

  private val mlpModel = FeedForwardTopology.multiLayerPerceptron(layers, true).model(weights)

  /**
   * Returns layers in a Java List.
   */
  private[ml] def javaLayers: java.util.List[Int] = {
    layers.toList.asJava
  }

  /**
   * Predict label for the given features.
   * This internal method is used to implement [[transform()]] and output [[predictionCol]].
   */
  override protected def predict(features: Vector): Double = {
    LabelConverter.decodeLabel(mlpModel.predict(features))
  }

  override def copy(extra: ParamMap): MultilayerPerceptronClassificationModel = {
    copyValues(new MultilayerPerceptronClassificationModel(uid, layers, weights), extra)
  }
}
