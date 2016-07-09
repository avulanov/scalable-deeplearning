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

package com.github.avulanov.scaleann

import java.util.Random

import com.github.avulanov.scaleann.optimization._

//import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, axpy => Baxpy}
import com.github.avulanov.tensor.DenseTensor
//import org.apache.spark.ml.ann.BreezeUtil
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom

object AnnTypes {
  type Tensor = DenseTensor[Double]
}

import AnnTypes._

/**
  * Trait that holds Layer properties, that are needed to instantiate it.
  * Implements Layer instantiation.
  *
  */
private[scaleann] trait Layer extends Serializable {

  /**
    * Number of weights that is used to allocate memory for the weights vector
    */
  val weightSize: Int

  /**
    * Returns the output size given the input size (not counting the stack size).
    * Output size is used to allocate memory for the output.
    *
    * @param inputSize input size
    * @return output size
    */
  def outputSize(inputSize: Int): Int

  /**
    * If true, the memory is not allocated for the output of this layer.
    * The memory allocated to the previous layer is used to write the output of this layer.
    * Developer can set this to true if computing delta of a previous layer
    * does not involve its output, so the current layer can write there.
    * This also mean that both layers have the same number of outputs.
    */
  val inPlace: Boolean

  /**
    * Returns the instance of the layer based on weights provided.
    * Size of weights must be equal to weightSize
    *
    * @param weights vector with layer weights
    * @return the layer model
    */
  def model(weights: Tensor): LayerModel
  /**
    * Returns the instance of the layer with random generated weights
    *
    * @param weights vector for weights initialization, must be equal to weightSize
    * @param random random number generator
    * @return the layer model
    */
  def initModel(weights: Tensor, random: Random): LayerModel
}

/**
  * Trait that holds Layer weights (or parameters).
  * Implements functions needed for forward propagation, computing delta and gradient.
  * Can return weights in Vector format.
  */
private[scaleann] trait LayerModel extends Serializable {

  val weights: Tensor
  /**
    * Evaluates the data (process the data through the layer)
    *
    * @param data data
    * @param output output to write to
    */
  def eval(data: Tensor, output: Tensor): Unit

  /**
    * Computes the delta for back propagation
    *
    * @param delta delta of this layer
    * @param output output of this layer
    * @param pDelta storage for the result, the previous delta
    * @return delta
    */
  def prevDelta(delta: Tensor, output: Tensor, pDelta: Tensor): Unit

  /**
    * Computes the gradient
    *
    * @param delta delta for this layer
    * @param input input data
    * @param cumGrad cumulative gradient
    * @return gradient
    */
  def grad(delta: Tensor, input: Tensor, cumGrad: Tensor): Unit
}

/**
  * Layer properties of affine transformations, that is y=A*x+b
  *
  * @param numIn number of inputs
  * @param numOut number of outputs
  */
private[scaleann] class AffineLayer(val numIn: Int, val numOut: Int) extends Layer {

  override val weightSize = numIn * numOut + numOut

  override def outputSize(inputSize: Int): Int = numOut

  override val inPlace = false

  override def model(weights: Tensor): LayerModel = new AffineLayerModel(weights, this)

  override def initModel(weights: Tensor, random: Random): LayerModel =
    AffineLayerModel(this, weights, random)
}

/**
  * Model of Affine layer
  *
  * @param weights weights
  * @param layer layer properties
  */
private[scaleann] class AffineLayerModel private[scaleann] (
                                                   val weights: Tensor,
                                                   val layer: AffineLayer) extends LayerModel {
//  val w = new BDM[Double](layer.numOut, layer.numIn, weights.data, weights.offset)
//  val b =
//    new BDV[Double](weights.data, weights.offset + (layer.numOut * layer.numIn), 1, layer.numOut)
  val w = DenseTensor(weights.data, Array(layer.numOut, layer.numIn), weights.offset)
  val b = DenseTensor(weights.data, Array(layer.numOut), weights.offset + (layer.numOut * layer.numIn))

  private var ones: Tensor = null

  override def eval(data: Tensor, output: Tensor): Unit = {
//    output(::, *) := b
//    BreezeUtil.dgemm(1.0, w, data, 1.0, output)
    output.fillWith(b)
    DenseTensor.gemm(1.0, w, data, 1.0, output)
  }

  override def prevDelta(nextDelta: Tensor, input: Tensor, delta: Tensor): Unit = {
    //BreezeUtil.dgemm(1.0, w.t, nextDelta, 0.0, delta)
    DenseTensor.gemm(1.0, w.transpose, nextDelta, 0.0, delta)
  }

  override def grad(delta: Tensor, input: Tensor, cumGrad: Tensor): Unit = {
    // compute gradient of weights
//    val cumGradientOfWeights = new BDM[Double](w.rows, w.cols, cumGrad.data, cumGrad.offset)
//    BreezeUtil.dgemm(1.0 / input.cols, delta, input.t, 1.0, cumGradientOfWeights)
//    if (ones == null || ones.length != delta.cols) ones = BDV.ones[Double](delta.cols)
    val cumGradientOfWeights = DenseTensor(cumGrad.data, w.shape, cumGrad.offset)
    DenseTensor.gemm(1.0 / input.shape(1), delta, input.transpose, 1.0, cumGradientOfWeights)
    if (ones == null || ones.shape(0) != delta.shape(1)) ones =
      DenseTensor.fill(Array(delta.shape(1)))(1)

    // compute gradient of bias
//    val cumGradientOfBias = new BDV[Double](cumGrad.data, cumGrad.offset + w.size, 1, b.length)
//    BreezeUtil.dgemv(1.0 / input.cols, delta, ones, 1.0, cumGradientOfBias)
    val cumGradientOfBias = DenseTensor(cumGrad.data, Array(b.shape(0)), cumGrad.offset + w.size)
    DenseTensor.gemv(1.0 / input.shape(1), delta, ones, 1.0, cumGradientOfBias)
  }
}

/**
  * Fabric for Affine layer models
  */
private[scaleann] object AffineLayerModel {

  /**
    * Creates a model of Affine layer
    *
    * @param layer layer properties
    * @param weights vector for weights initialization
    * @param random random number generator
    * @return model of Affine layer
    */
  def apply(layer: AffineLayer, weights: Tensor, random: Random): AffineLayerModel = {
    randomWeights(layer.numIn, layer.numOut, weights, random)
    new AffineLayerModel(weights, layer)
  }

  /**
    * Initialize weights
    *
    * @param numIn number of inputs
    * @param numOut number of outputs
    * @param weights vector for weights initialization
    * @param random random number generator
    */
  def randomWeights(
                     numIn: Int,
                     numOut: Int,
                     weights: Tensor,
                     random: Random): Unit = {
    var i = 0
    val sz = weights.size
    while (i < sz) {
      //weights(i) = (random.nextDouble * 4.8 - 2.4) / numIn
      weights.update(i, (random.nextDouble * 4.8 - 2.4) / numIn)
      i += 1
    }
  }
}

/**
  * Trait for functions and their derivatives for functional layers
  */
private[scaleann] trait ActivationFunction extends Serializable {

  /**
    * Implements a function
    */
  def eval: Double => Double

  /**
    * Implements a derivative of a function (needed for the back propagation)
    */
  def derivative: Double => Double
}

///**
//  * Implements in-place application of functions in the arrays
//  */
//private[anntensor] object UniversalFunction {
//
//  // TODO: use Breeze UFunc
//  def apply(x: BDM[Double], y: BDM[Double], func: Double => Double): Unit = {
//    var i = 0
//    while (i < x.rows) {
//      var j = 0
//      while (j < x.cols) {
//        y(i, j) = func(x(i, j))
//        j += 1
//      }
//      i += 1
//    }
//  }
//
//  // TODO: use Breeze UFunc
//  def apply(
//             x1: BDM[Double],
//             x2: BDM[Double],
//             y: BDM[Double],
//             func: (Double, Double) => Double): Unit = {
//    var i = 0
//    while (i < x1.rows) {
//      var j = 0
//      while (j < x1.cols) {
//        y(i, j) = func(x1(i, j), x2(i, j))
//        j += 1
//      }
//      i += 1
//    }
//  }
//}

/**
  * Implements Sigmoid activation function
  */
private[scaleann] class SigmoidFunction extends ActivationFunction {

  override def eval: (Double) => Double = x => 1.0 / (1 + Math.exp(-x))

  override def derivative: (Double) => Double = z => (1 - z) * z
}

/**
  * Functional layer properties, y = f(x)
  *
  * @param activationFunction activation function
  */
private[scaleann] class FunctionalLayer (val activationFunction: ActivationFunction) extends Layer {

  override val weightSize = 0

  override def outputSize(inputSize: Int): Int = inputSize

  override val inPlace = true

  override def model(weights: Tensor): LayerModel = new FunctionalLayerModel(this)

  override def initModel(weights:Tensor, random: Random): LayerModel =
    model(weights)
}

/**
  * Functional layer model. Holds no weights.
  *
  * @param layer functiona layer
  */
private[scaleann] class FunctionalLayerModel private[scaleann] (val layer: FunctionalLayer)
  extends LayerModel {

  // empty weights
  val weights: Tensor = DenseTensor(Array(0))

  override def eval(data: Tensor, output: Tensor): Unit = {
    //UniversalFunction(data, output, layer.activationFunction.eval)
    DenseTensor.applyFunction(data, output, layer.activationFunction.eval)
  }

  override def prevDelta(nextDelta: Tensor, input: Tensor, delta: Tensor): Unit = {
//    UniversalFunction(input, delta, layer.activationFunction.derivative)
//    delta :*= nextDelta
    DenseTensor.applyFunction(input, delta, layer.activationFunction.derivative)
    DenseTensor.elementwiseProduct(delta, nextDelta)
  }

  override def grad(delta: Tensor, input: Tensor, cumGrad: Tensor): Unit = {}
}

/**
  * Trait for the artificial neural network (ANN) topology properties
  */
private[scaleann] trait Topology extends Serializable {
  def model(weights: Vector): TopologyModel
  def model(seed: Long): TopologyModel
}

/**
  * Trait for ANN topology model
  */
private[scaleann] trait TopologyModel extends Serializable {

  val weights: Vector
  /**
    * Array of layers
    */
  val layers: Array[Layer]

  /**
    * Array of layer models
    */
  val layerModels: Array[LayerModel]
  /**
    * Forward propagation
    *
    * @param data input data
    * @return array of outputs for each of the layers
    */
  def forward(data: Tensor): Array[Tensor]

  /**
    * Prediction of the model
    *
    * @param data input data
    * @return prediction
    */
  def predict(data: Vector): Vector

  /**
    * Computes gradient for the network
    *
    * @param data input data
    * @param target target output
    * @param cumGradient cumulative gradient
    * @param blockSize block size
    * @return error
    */
  def computeGradient(data: Tensor, target: Tensor, cumGradient: Tensor,
                      blockSize: Int): Double
}

/**
  * Feed forward ANN
  *
  * @param layers
  */
private[scaleann] class FeedForwardTopology private(val layers: Array[Layer]) extends Topology {
  override def model(weights: Vector): TopologyModel = FeedForwardModel(this, weights)

  override def model(seed: Long): TopologyModel = FeedForwardModel(this, seed)
}

/**
  * Factory for some of the frequently-used topologies
  */
object FeedForwardTopology {
  /**
    * Creates a feed forward topology from the array of layers
    *
    * @param layers array of layers
    * @return feed forward topology
    */
  def apply(layers: Array[Layer]): FeedForwardTopology = {
    new FeedForwardTopology(layers)
  }

  /**
    * Creates a multi-layer perceptron
    *
    * @param layerSizes sizes of layers including input and output size
    * @param softmaxOnTop wether to use SoftMax or Sigmoid function for an output layer.
    *                Softmax is default
    * @return multilayer perceptron topology
    */
  def multiLayerPerceptron(
                            layerSizes: Array[Int],
                            softmaxOnTop: Boolean = true): FeedForwardTopology = {
    val layers = new Array[Layer]((layerSizes.length - 1) * 2)
    for(i <- 0 until layerSizes.length - 1){
      layers(i * 2) = new AffineLayer(layerSizes(i), layerSizes(i + 1))
      layers(i * 2 + 1) =
        if (i == layerSizes.length - 2) {
          if (softmaxOnTop) {
            new SoftmaxLayerWithCrossEntropyLoss()
          } else {
            // TODO: squared error is more natural but converges slower
            new SigmoidLayerWithSquaredError()
          }
        } else {
          new FunctionalLayer(new SigmoidFunction())
        }
    }
    FeedForwardTopology(layers)
  }
}

/**
  * Model of Feed Forward Neural Network.
  * Implements forward, gradient computation and can return weights in vector format.
  *
  * @param weights network weights
  * @param topology network topology
  */
class FeedForwardModel private(
                                            val weights: Vector,
                                            val topology: FeedForwardTopology) extends TopologyModel {

  val layers = topology.layers
  val layerModels = new Array[LayerModel](layers.length)
  private var offset = 0
  for (i <- 0 until layers.length) {
    layerModels(i) = layers(i).model(
      //new BDV[Double](weights.toArray, offset, 1, layers(i).weightSize))
     DenseTensor(weights.toArray, Array(layers(i).weightSize), offset))
    offset += layers(i).weightSize
  }
  private var outputs: Array[Tensor] = null
  private var deltas: Array[Tensor] = null

  override def forward(data: Tensor): Array[Tensor] = {
    // Initialize output arrays for all layers. Special treatment for InPlace
    val currentBatchSize = data.shape(1)
    // TODO: allocate outputs as one big array and then create BDMs from it
    if (outputs == null || outputs(0).shape(1) != currentBatchSize) {
      outputs = new Array[Tensor](layers.length)
      var inputSize = data.shape(0)
      for (i <- 0 until layers.length) {
        if (layers(i).inPlace) {
          outputs(i) = outputs(i - 1)
        } else {
          val outputSize = layers(i).outputSize(inputSize)
          //outputs(i) = new BDM[Double](outputSize, currentBatchSize)
          outputs(i) = DenseTensor(Array(outputSize, currentBatchSize))
          inputSize = outputSize
        }
      }
    }
    layerModels(0).eval(data, outputs(0))
    for (i <- 1 until layerModels.length) {
      layerModels(i).eval(outputs(i - 1), outputs(i))
    }
    outputs
  }

  override def computeGradient(
                                data: Tensor,
                                target: Tensor,
                                cumGradient: Tensor,
                                realBatchSize: Int): Double = {
    val outputs = forward(data)
    val currentBatchSize = data.shape(1)
    // TODO: allocate deltas as one big array and then create BDMs from it
    if (deltas == null || deltas(0).shape(1) != currentBatchSize) {
      deltas = new Array[Tensor](layerModels.length)
      var inputSize = data.shape(0)
      for (i <- 0 until layerModels.length - 1) {
        val outputSize = layers(i).outputSize(inputSize)
        //deltas(i) = new BDM[Double](outputSize, currentBatchSize)
        deltas(i) = DenseTensor(Array(outputSize, currentBatchSize))
        inputSize = outputSize
      }
    }
    val L = layerModels.length - 1
    // TODO: explain why delta of top layer is null (because it might contain loss+layer)
    val loss = layerModels.last match {
      case levelWithError: LossFunction => levelWithError.loss(outputs.last, target, deltas(L - 1))
      case _ =>
        throw new UnsupportedOperationException("Top layer is required to have objective.")
    }
    for (i <- (L - 2) to (0, -1)) {
      layerModels(i + 1).prevDelta(deltas(i + 1), outputs(i + 1), deltas(i))
    }
    val cumGradientArray = cumGradient.data
    var offset = 0
    for (i <- 0 until layerModels.length) {
      val input = if (i == 0) data else outputs(i - 1)
      layerModels(i).grad(deltas(i), input,
        //new BDV[Double](cumGradientArray, offset, 1, layers(i).weightSize))
        new Tensor(cumGradientArray, Array(layers(i).weightSize), offset))
      offset += layers(i).weightSize
    }
    loss
  }

  override def predict(data: Vector): Vector = {
    val size = data.size
//    val result = forward(new BDM[Double](size, 1, data.toArray))
//    Vectors.dense(result.last.toArray)
    val result = forward(DenseTensor(data.toArray, Array(size, 1)))
    // TODO: check that it was OK not to clone in the previous version
    Vectors.dense(result.last.data.clone())
  }
}

/**
  * Fabric for feed forward ANN models
  */
private[scaleann] object FeedForwardModel {

  /**
    * Creates a model from a topology and weights
    *
    * @param topology topology
    * @param weights weights
    * @return model
    */
  def apply(topology: FeedForwardTopology, weights: Vector): FeedForwardModel = {
    // TODO: check that weights size is equal to sum of layers sizes
    new FeedForwardModel(weights, topology)
  }

  /**
    * Creates a model given a topology and seed
    *
    * @param topology topology
    * @param seed seed for generating the weights
    * @return model
    */
  def apply(topology: FeedForwardTopology, seed: Long = 11L): FeedForwardModel = {
    val layers = topology.layers
    val layerModels = new Array[LayerModel](layers.length)
    var totalSize = 0
    for (i <- 0 until topology.layers.length) {
      totalSize += topology.layers(i).weightSize
    }
    //val weights = new BDV[Double](new Array[Double](totalSize))
    val weights: Tensor = DenseTensor(Array(totalSize))
    var offset = 0
    // TODO: check if we can re-use XORShiftRandom
    val random = new Random(seed)
    for(i <- 0 until layers.length){
      layerModels(i) = layers(i).
        initModel(DenseTensor(weights.data, Array(layers(i).weightSize), offset), random)
        //initModel(new BDV[Double](weights.data, offset, 1, layers(i).weightSize), random)
      offset += layers(i).weightSize
    }
    //new FeedForwardModel(Vectors.fromBreeze(weights), topology)
    new FeedForwardModel(Vectors.dense(weights.data), topology)
  }
}

/**
  * Neural network gradient. Does nothing but calling Model's gradient
  *
  * @param topology topology
  * @param dataStacker data stacker
  */
private[scaleann] class ANNGradient(topology: Topology, dataStacker: DataStacker) extends Gradient {

  override def compute(data: Vector, label: Double, weights: Tensor): (Tensor, Double) = {
    val gradient = new Tensor(Array(weights.size))
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(
                        data: Vector,
                        label: Double,
                        weights: Tensor,
                        cumGradient: Tensor): Double = {
    val (input, target, realBatchSize) = dataStacker.unstack(data)
    val model = topology.model(Vectors.dense(weights.data))
    model.computeGradient(input, target, cumGradient, realBatchSize)
  }
}

/**
  * Stacks pairs of training samples (input, output) in one vector allowing them to pass
  * through Optimizer/Gradient interfaces. If stackSize is more than one, makes blocks
  * or matrices of inputs and outputs and then stack them in one vector.
  * This can be used for further batch computations after unstacking.
  *
  * @param stackSize stack size
  * @param inputSize size of the input vectors
  * @param outputSize size of the output vectors
  */
private[scaleann] class DataStacker(stackSize: Int, inputSize: Int, outputSize: Int)
  extends Serializable {

  /**
    * Stacks the data
    *
    * @param data RDD of vector pairs
    * @return RDD of double (always zero) and vector that contains the stacked vectors
    */
  def stack(data: RDD[(Vector, Vector)]): RDD[(Double, Vector)] = {
    val stackedData = if (stackSize == 1) {
      data.map { v =>
        val bigVector = new Array[Double](v._1.size + v._2.size)
        System.arraycopy(v._1.toArray, 0, bigVector, 0, v._1.size)
        System.arraycopy(v._2.toArray, 0, bigVector, v._1.size, v._2.size)
        (0.0, Vectors.dense(bigVector))
      }
    } else {
      data.mapPartitions { it =>
        it.grouped(stackSize).map { seq =>
          val size = seq.size
          val bigVector = new Array[Double](inputSize * size + outputSize * size)
          var i = 0
          seq.foreach { case (in, out) =>
            System.arraycopy(in.toArray, 0, bigVector, i * inputSize, inputSize)
            System.arraycopy(out.toArray, 0, bigVector,
              inputSize * size + i * outputSize, outputSize)
            i += 1
          }
          (0.0, Vectors.dense(bigVector))
        }
      }
    }
    stackedData
  }

  /**
    * Unstack the stacked vectors into matrices for batch operations
    *
    * @param data stacked vector
    * @return pair of matrices holding input and output data and the real stack size
    */
  def unstack(data: Vector): (Tensor, Tensor, Int) = {
    val arrData = data.toArray
    val realStackSize = arrData.length / (inputSize + outputSize)
//    val input = new BDM(inputSize, realStackSize, arrData)
//    val target = new BDM(outputSize, realStackSize, arrData, inputSize * realStackSize)
    val input =DenseTensor(arrData, Array(inputSize, realStackSize))
    val target = DenseTensor(arrData, Array(outputSize, realStackSize), inputSize * realStackSize)
    (input, target, realStackSize)
  }
}

/**
  * Simple updater
  */
private[scaleann] class ANNUpdater extends Updater {

  override def compute(
                        weightsOld: Tensor,
                        gradient: Tensor,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Tensor, Double) = {
    val thisIterStepSize = stepSize
//    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
//    Baxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    DenseTensor.axpy(-thisIterStepSize, gradient, weightsOld)
    (weightsOld, 0)
  }
}

/**
  * MLlib-style trainer class that trains a network given the data and topology
  *
  * @param topology topology of ANN
  * @param inputSize input size
  * @param outputSize output size
  */
class FeedForwardTrainer(
                                      topology: Topology,
                                      val inputSize: Int,
                                      val outputSize: Int) extends Serializable {

  private var _seed = 11L
  private var _weights: Vector = null
  private var _stackSize = 128
  private var dataStacker = new DataStacker(_stackSize, inputSize, outputSize)
  private var _gradient: Gradient = new ANNGradient(topology, dataStacker)
  private var _updater: Updater = new ANNUpdater()
  private var optimizer: Optimizer = LBFGSOptimizer.setConvergenceTol(1e-4).setNumIterations(100)

  /**
    * Returns seed
    *
    * @return seed
    */
  def getSeed: Long = _seed

  /**
    * Sets seed
    *
    * @param value seed
    * @return trainer
    */
  def setSeed(value: Long): FeedForwardTrainer = {
    _seed = value
    this
  }

  /**
    * Returns weights
    *
    * @return weights
    */
  def getWeights: Vector = _weights

  /**
    * Sets weights
    *
    * @param value weights
    * @return trainer
    */
  def setWeights(value: Vector): FeedForwardTrainer = {
    _weights = value
    this
  }

  /**
    * Sets the stack size
    *
    * @param value stack size
    * @return trainer
    */
  def setStackSize(value: Int): FeedForwardTrainer = {
    _stackSize = value
    dataStacker = new DataStacker(value, inputSize, outputSize)
    this
  }

  /**
    * Sets the SGD optimizer
    *
    * @return SGD optimizer
    */
  def SGDOptimizer: GradientDescent = {
    val sgd = new GradientDescent(_gradient, _updater)
    optimizer = sgd
    sgd
  }

  /**
    * Sets the LBFGS optimizer
    *
    * @return LBGS optimizer
    */
  def LBFGSOptimizer: LBFGS = {
    val lbfgs = new LBFGS(_gradient, _updater)
    optimizer = lbfgs
    lbfgs
  }

  /**
    * Sets the updater
    *
    * @param value updater
    * @return trainer
    */
  def setUpdater(value: Updater): FeedForwardTrainer = {
    _updater = value
    updateUpdater(value)
    this
  }

  /**
    * Sets the gradient
    *
    * @param value gradient
    * @return trainer
    */
  def setGradient(value: Gradient): FeedForwardTrainer = {
    _gradient = value
    updateGradient(value)
    this
  }

  private[this] def updateGradient(gradient: Gradient): Unit = {
    optimizer match {
      case lbfgs: LBFGS => lbfgs.setGradient(gradient)
      case sgd: GradientDescent => sgd.setGradient(gradient)
      case other => throw new UnsupportedOperationException(
        s"Only LBFGS and GradientDescent are supported but got ${other.getClass}.")
    }
  }

  private[this] def updateUpdater(updater: Updater): Unit = {
    optimizer match {
      case lbfgs: LBFGS => lbfgs.setUpdater(updater)
      case sgd: GradientDescent => sgd.setUpdater(updater)
      case other => throw new UnsupportedOperationException(
        s"Only LBFGS and GradientDescent are supported but got ${other.getClass}.")
    }
  }

  /**
    * Trains the ANN
    *
    * @param data RDD of input and output vector pairs
    * @return model
    */
  def train(data: RDD[(Vector, Vector)]): TopologyModel = {
    val w = if (getWeights == null) {
      // TODO: will make a copy if vector is a subvector of BDV (see Vectors code)
      topology.model(_seed).weights
    } else {
      getWeights
    }
    // TODO: deprecate standard optimizer because it needs Vector
    val newWeights = optimizer.optimize(dataStacker.stack(data), new Tensor(w.toArray, Array(w.size), 0))
    topology.model(Vectors.dense(newWeights.data))
  }

}
