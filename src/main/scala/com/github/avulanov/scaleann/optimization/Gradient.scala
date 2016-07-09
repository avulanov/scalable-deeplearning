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

package com.github.avulanov.scaleann.optimization

import com.github.avulanov.scaleann.AnnTypes.Tensor
import com.github.avulanov.tensor.DenseTensor
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
//import org.apache.spark.mllib.linalg.BLAS.{axpy, dot, scal}
//import org.apache.spark.mllib.util.MLUtils

/**
 * :: DeveloperApi ::
 * Class used to compute the gradient for a loss function, given a single data point.
 */
@DeveloperApi
abstract class Gradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
    * @return (gradient: Vector, loss: Double)
   */
  def compute(data: Vector, label: Double, weights: Tensor): (Tensor, Double) = {
    val gradient = new Tensor(Array(weights.size))
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  /**
   * Compute the gradient and loss given the features of a single data point,
   * add the gradient to a provided vector to avoid creating new objects, and return loss.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   * @param cumGradient the computed gradient will be added to this vector
    * @return loss
   */
  def compute(data: Vector, label: Double, weights: Tensor, cumGradient: Tensor): Double
}
