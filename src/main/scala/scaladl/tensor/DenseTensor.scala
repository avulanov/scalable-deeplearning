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

package scaladl.tensor

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

object Algebra {
  trait NumberLike[@specialized (Double, Float) T] extends Serializable {
    def plus(x: T, y: T): T
    def minus(x: T, y: T): T
    def times(x: T, y: T): T
    def sqrt(x: T): T
  }
  object NumberLike {
    implicit object NumberLikeDouble extends NumberLike[Double] {
      def plus(x: Double, y: Double): Double = x + y
      def minus(x: Double, y: Double): Double = x - y
      def times(x: Double, y: Double): Double = x * y
      def sqrt(x: Double): Double = math.sqrt(x)
    }
    implicit object NumberLikeFloat extends NumberLike[Float] {
      def plus(x: Float, y: Float): Float = x + y
      def minus(x: Float, y: Float): Float = x - y
      def times(x: Float, y: Float): Float = x * y
      def sqrt(x: Float): Float = math.sqrt(x.toDouble).toFloat
    }
  }
}
  import Algebra.NumberLike

/**
 * Dense tensor column-major representation. // TODO: row major??
 *
 * @param data underlying data
 * @param tensorShape shape of tensor
 * @param offset offset in the data
 * @tparam T type
 */
class DenseTensor[@specialized(Double, Float) T] (
  val data: Array[T],
  val tensorShape: Array[Int],
  val offset: Int,
  isTransposed: Boolean = false)(implicit numOps: NumberLike[T]) extends Serializable {

  private var actualSize: Int = 0
  private var majorStride: Int = 0
  private var requiredSize: Int = 0
  // Fix of the Scala specialized constructor bug:
  // http://axel22.github.io/2013/11/03/specialization-quirks.html
  protected def init(data: Array[T], tensorShape: Array[Int]): Unit = {
    actualSize = data.length - offset
    majorStride = if (isTransposed) tensorShape.last else tensorShape.head
    requiredSize = tensorShape.product
  }
  init(data, tensorShape)
  // TODO: figure out which of size, shape etc can be removed or replaced in other functions
//  private val actualSize = data.size//data.length - offset
//  // Major stride (always the first??? dimension since stored in columnar format)
//  private val majorStride = if (isTransposed) tensorShape.last else tensorShape.head
//  private val requiredSize = tensorShape.product
  require(requiredSize <= actualSize,
    "Actual size of the array does not correspond to dimension Sizes")
  private var myShape = tensorShape

  /**
   * Allocate new tensor
   * @param tensorShape tensor shape
   * @param m type parameter
   * @param numOps ops parameter
   */
  def this(tensorShape: Array[Int])(implicit m: ClassTag[T], numOps: NumberLike[T]) = {
    this(new Array[T](tensorShape.product), tensorShape, 0)
  }

  /**
   * New tensor given data and shape
   * @param data data array
   * @param tensorShape shape
   * @param m type
   * @param numOps ops
   */
  def this(data: Array[T], tensorShape: Array[Int])
          (implicit m: ClassTag[T], numOps: NumberLike[T]) = {
    this(data, tensorShape, 0, false)
  }

  /**
   * Don't use this in loops!!!
   * @return
   */
  def size: Int = myShape.product
  /**
   * Shape of the tensor
   *
   * @return shape
   */
  def shape: Array[Int] = myShape

  /**
   * Reshape the tensor. Supports reshaping within the same data size
   *
   * @param newShape new shape
   * @return reshaped tensor backed by the same data
   */
  def reshape(newShape: Array[Int]): DenseTensor[T] = {
    val requiredSize = newShape.product
    require(requiredSize == actualSize)
    myShape = newShape
    this
  }

  /**
   * Update value of a Tensor
   *
   * @param index index
   * @param value value
   */
  def update(index: Int, value: T): Unit = {
    require(index >=0 && index < requiredSize)
    data(this.offset + index) = value
  }

  /**
   * Update value of a Tensor
   *
   * @param index index
   * @param value value
   */
  def update(index: Array[Int], value: T): Unit = {
    data(offset(index)) = value
  }

  /**
   * Get the value at position index
   *
   * @param index index
   * @return value
   */
  def value(index: Int): T = {
    require(index >=0 && index < requiredSize)
    data(this.offset + index)
  }

  /**
   * Get the value at position index
   *
   * @param index index
   * @return value
   */
  def value(index: Array[Int]): T = {
    data(offset(index))
  }

  private def offset(index: Array[Int]): Int = {
    var offset = index.last
    for (i <- myShape.length - 1 to 1 by -1) {
      offset = index(i - 1) + myShape(i - 1) * offset
    }
    offset
  }

  /**
   * Check if tensor is transposed
   *
   * @return true if transposed, false otherwise
   */
  def transposed: Boolean = isTransposed

  /**
   * Transpose tensor. Does not actually transpose the data.
   * It is used for operations such as gemm.
   *
   * @return self
   */
  def transpose(implicit m: ClassTag[T]): DenseTensor[T] = {
    require(tensorShape.length == 2, "Transpose is valid only for 2 dimensional tensor")
    val transposedTensor = DenseTensor[T](data, tensorShape.reverse, offset, true)
    transposedTensor
  }

  /**
   * Slice the tensor by the last dimension
   *
   * @param from index
   * @param until index
   * @return tensor backed by the same data
   */
  def slice(from: Int, until: Int): DenseTensor[T] = {
    require(from < until && from < myShape(0) && until <= myShape(0),
      "start and end must be within the size of first dimension, also start <= end")
    val shapeInit = myShape.init
    val lastDimensionNewSize = until - from
    val startOffset = offset(shapeInit.map(_ => 0) :+ from)
    new DenseTensor[T](data, shapeInit :+ lastDimensionNewSize, startOffset)
  }

  /**
   * Slice the tensor by one index in the last dimension
   *
   * @param index index
   * @return squeezed tensor
   */
  def slice(index: Int): DenseTensor[T] = {
    slice(index, index + 1).squeeze()
  }

  /**
   * Squeze the dimensions of size 1
   *
   * @return tensor backed by the same data
   */
  def squeeze(): DenseTensor[T] = {
    val buf = new ArrayBuffer[Int](myShape.length)
    for (dim <- myShape) {
      if (dim > 1) buf += dim
    }
    myShape = buf.toArray
    this
  }

  /**
   * Copy the underlying data
   *
   * @param m ClassTag
   * @return data array
   */
  def copyData()(implicit m: ClassTag[T]): Array[T] = {
    val array = new Array[T](myShape.product)
    System.arraycopy(data, offset, array, 0, array.length)
    array
  }

  def copy()(implicit m: ClassTag[T]): DenseTensor[T] = {
    val array = new Array[T](myShape.product)
    System.arraycopy(data, offset, array, 0, array.length)
    new DenseTensor(array, myShape, offset, isTransposed)
  }

  /**
   * Fill tensor with the data from the other tensor
   *
   * @param donor tensor from which to get data
   * @return self
   */
  def fillWith(donor: DenseTensor[T]): DenseTensor[T] = {
    require(size % donor.size == 0 && size >= donor.size,
      "data size of recipient tensor must be >= and divide evenly by the data size of donor tensor")
    val donorSize = donor.size
    val numCopies = size / donorSize
    var k = 0
    var nextOffset = 0
    while (k <  numCopies) {
      System.arraycopy(donor.data, donor.offset, this.data, this.offset + nextOffset, donorSize)
      nextOffset += donorSize
      k += 1
    }
    this
  }

  /**
   * Plus operation
   * @param other other tensor
   * @param m type parameter
   * @return returns new tensor
   */
  def +(other: DenseTensor[T])(implicit m: ClassTag[T]): DenseTensor[T] = {
    require(equalShape(other), "Must be equal shape")
    val sz = size
    val newData = new Array[T](sz)
    var thisIndex = this.offset
    var otherIndex = other.offset
    var i = 0
    while (i < sz) {
      newData(i) = numOps.plus(this.data(thisIndex), other.data(otherIndex))
      thisIndex += 1
      otherIndex += 1
      i += 1
    }
    DenseTensor(newData, shape.clone())
  }

  /**
   * Minus operation
   * @param other other tensor
   * @param m type parameter
   * @return returns new tensor
   */
  def -(other: DenseTensor[T])(implicit m: ClassTag[T]): DenseTensor[T] = {
    require(equalShape(other), "Must be equal shape")
    val sz = size
    val newData = new Array[T](sz)
    var i = 0
    while (i < sz) {
      newData(i) = numOps.minus(this.data(this.offset + i), other.data(other.offset + i))
      i += 1
    }
    DenseTensor(newData, shape.clone())
  }

  /**
   * Elementwise multiplication
   * @param other other tensor
   * @param m type parameter
   * @return returns new tensor
   */
  def :*(other: DenseTensor[T])(implicit m: ClassTag[T]): DenseTensor[T] = {
    require(equalShape(other), "Must be equal shape")
    val sz = size
    val newData = new Array[T](sz)
    var i = 0
    while (i < sz) {
      newData(i) = numOps.times(this.data(this.offset + i), other.data(other.offset + i))
      i += 1
    }
    DenseTensor(newData, shape.clone())
  }

  // TODO: fix this ugliness
  private def equalShape(other: DenseTensor[T]): Boolean = {
    val thisShape = this.shape
    val otherShape = other.shape
    if (thisShape.length != otherShape.length) {
      return false
    } else {
      var i = 0
      while (i < thisShape.length) {
        if (thisShape(i) != otherShape(i)) {
          return false
        }
        i += 1
      }
    }
    true
  }

  /**
   * Sum of the elements
   * @return sum
   */
  def sum: T = {
    var i = offset
    var mySum = numOps.minus(data(i), data(i))
    val max = offset + size
    while (i < max) {
      mySum = numOps.plus(mySum, data(i))
      i += 1
    }
    mySum
  }

  /**
   * Norm of the vector
   * @return norm
   */
  def norm: T = {
    var i = offset
    var mySum = numOps.minus(data(i), data(i))
    val max = offset + size
    while (i < max) {
      mySum = numOps.plus(mySum, numOps.times(data(i), data(i)))
      i += 1
    }
    numOps.sqrt(mySum)
  }

  /**
   * Equals for transposed, shape and data
   * @param other tensor
   * @return true if equal, false overwise
   */
  def isEqual(other: DenseTensor[T]): Boolean = {
    if (this.transposed != other.transposed || !equalShape(other)) {
      return false
    } else {
      var i = 0
      while (i < data.length) {
        if (data(i) != other.data(i)) {
          return false
        }
        i += 1
      }
    }
    true
  }

  override def toString(): String = {
    // TODO: implement row-by-row print
    val buf = new StringBuilder()
    for (i <- offset until offset + size) {
      var product: Int = 1
      val index = new Array[Int](myShape.length)
      for (dim <- 0 until myShape.length - 1) {
        val dimValue = (i / product) % myShape(dim)
        product *= myShape(dim)
        index(dim) = dimValue
      }
      index(myShape.length - 1) = i / product
      buf.append(value(index))
      buf.append(" ")
    }
    buf.toString()
  }

}

object DenseTensor {

  /**
   * Create a tensor with zeros
   *
   * @param tensorShape shape
   * @param m ClassTag
   * @tparam T implicit type
   * @return tensor
   */
  def apply[@specialized(Double, Float) T](tensorShape: Array[Int])
                                          (implicit m: ClassTag[T],
                                           numOps: NumberLike[T]): DenseTensor[T] = {
    val data: Array[T] = new Array[T](tensorShape.product)
    DenseTensor(data, tensorShape)
  }

  /**
   * Create a tensor from data
   *
   * @param data data
   * @param tensorShape shape
   * @param offset offset in the data
   * @param m ClassTag
   * @tparam T implicit type
   * @return tensor
   */
  def apply[@specialized(Double, Float) T](
  data: Array[T],
  tensorShape: Array[Int],
  offset: Int = 0,
  isTransposed: Boolean = false)
  (implicit m: ClassTag[T], numOps: NumberLike[T]): DenseTensor[T] = {
    new DenseTensor[T](data, tensorShape, offset, isTransposed)
  }

  /**
   * Create and fill tensor with values
   *
   * @param tensorShape shape
   * @param elem value
   * @param m ClassTag
   * @tparam T type
   * @return tensor
   */
  def fill[@specialized(Double, Float) T](tensorShape: Array[Int])
                                         (elem: => T)
                                         (implicit m: ClassTag[T],
                                          numOps: NumberLike[T]): DenseTensor[T] = {
    val data: Array[T] = Array.fill[T](tensorShape.product)(elem)
    DenseTensor(data, tensorShape)
  }

  /**
   * Apply a function to tensor x in place
   *
   * @param x source
   * @param func function
   * @tparam T type
   */
  def applyFunction[@specialized(Double, Float) T](x: DenseTensor[T], func: T => T)
                      (implicit m: ClassTag[T], numOps: NumberLike[T]): Unit = {
    var i = x.offset
    val sz = x.offset + x.size
    while (i < sz) {
      x.data(i) = func(x.data(i))
      i += 1
    }
  }

  /**
   * Apply a function to tensor x and put the result in the y
   *
   * @param x source
   * @param y result
   * @param func function
   * @tparam T type
   */
  def applyFunction[@specialized(Double, Float) T](x: DenseTensor[T],
                                                   y: DenseTensor[T],
                                                   func: T => T)
                                                  (implicit m: ClassTag[T],
                                                   numOps: NumberLike[T]): Unit = {
    require(x.size == y.size, "Tensor sizes must be equal")
    var i = 0
    val sz = y.size
    while (i < sz) {
      y.data(y.offset + i) = func(x.data(x.offset + i))
      i += 1
    }
  }

  /**
   * Apply a function to tensor x and put the result in the y
   *
   * @param x1 source1
   * @param x2 source2
   * @param y result
   * @param func function
   * @tparam T type
   */
  def applyFunction[@specialized(Double, Float) T](
  x1: DenseTensor[T],
  x2: DenseTensor[T],
  y: DenseTensor[T],
  func: (T, T) => T)(implicit m: ClassTag[T], numOps: NumberLike[T]): Unit = {
    require(x1.size == y.size && x2.size == y.size, "Tensor sizes must be equal")
    var i = 0
    val sz = y.offset + y.size
    while (i < sz) {
      y.data(y.offset + i) = func(x1.data(x1.offset + i), x2.data(x2.offset + i))
      i += 1
    }
  }

  /**
   * Double 2d tensor multiplication C <- alpha * A * B + beta * C
   *
   * @param alpha alpha
   * @param a A
   * @param b B
   * @param beta beta
   * @param c C
   */
  def gemm(
  alpha: Double,
  a: DenseTensor[Double],
  b: DenseTensor[Double],
  beta: Double,
  c: DenseTensor[Double]): Unit = {
    // TODO: case with 3d and more
    require(a.shape.length == 2 && b.shape.length == 2 && c.shape.length == 2,
      "A, B, or C are not 2d tensors")
    // TODO: add code if matrices isTranspose!!!
    require(a.shape(1) == b.shape(0), "A & B Dimension mismatch!")
    require(a.shape(0) == c.shape(0), "A & C Dimension mismatch!")
    require(b.shape(1) == c.shape(1), "B & C Dimension mismatch!")
    NativeBLAS.dgemm(transposeString(a), transposeString(b), c.shape(0), c.shape(1), a.shape(1),
    // TODO: check majorStride
      alpha, a.data, a.offset, a.majorStride,
      b.data, b.offset, b.majorStride,
      beta, c.data, c.offset, c.shape(0))
  }

  /**
   * Double 2d tensor multiplication C <- alpha * A * B + beta * C
   *
   * @param alpha alpha
   * @param a A
   * @param b B
   * @param beta beta
   * @param c C
   */
  def gemm(
             alpha: Float,
             a: DenseTensor[Float],
             b: DenseTensor[Float],
             beta: Float,
             c: DenseTensor[Float]): Unit = {
    // TODO: case with 3d and more
    require(a.shape.length == 2 && b.shape.length == 2 && c.shape.length == 2,
      "A, B, or C are not 2d tensors")
    // TODO: add code if matrices isTranspose!!!
    require(a.shape(1) == b.shape(0), "A & B Dimension mismatch!")
    require(a.shape(0) == c.shape(0), "A & C Dimension mismatch!")
    require(b.shape(1) == c.shape(1), "B & C Dimension mismatch!")
    NativeBLAS.sgemm(transposeString(a), transposeString(b), c.shape(0), c.shape(1), a.shape(1),
      // TODO: check majorStride
      alpha, a.data, a.offset, a.majorStride,
      b.data, b.offset, b.majorStride,
      beta, c.data, c.offset, c.shape(0))
  }

  private def transposeString[T](a: DenseTensor[T]): String = if (a.transposed) "T" else "N"

  /**
   * GEMV: y := alpha * A * x + beta * y
   *
   * @param alpha alpha
   * @param a A
   * @param x x
   * @param beta beta
   * @param y y
   */
  def gemv(
  alpha: Double,
  a: DenseTensor[Double],
  x: DenseTensor[Double],
  beta: Double,
  y: DenseTensor[Double]): Unit = {
    require(a.shape.length == 2 && x.shape.length == 1 && y.shape.length == 1,
      "A must be 2d and X, Y - 1d tensors")
    require(a.shape(1) == x.shape(0), "A & X Dimension mismatch!")
    require(a.shape(0) == y.shape(0), "A & Y Dimension mismatch!")
    NativeBLAS.dgemv(transposeString(a), a.shape(0), a.shape(1),
      alpha, a.data, a.offset, a.shape(0),
      x.data, x.offset, 1,
      beta, y.data, y.offset, 1)
  }

  /**
   * GEMV: y := alpha * A * x + beta * y
   *
   * @param alpha alpha
   * @param a A
   * @param x x
   * @param beta beta
   * @param y y
   */
  def gemv(
  alpha: Float,
  a: DenseTensor[Float],
  x: DenseTensor[Float],
  beta: Float,
  y: DenseTensor[Float]): Unit = {
    require(a.shape.length == 2 && x.shape.length == 1 && y.shape.length == 1,
      "A must be 2d and X, Y - 1d tensors")
    require(a.shape(1) == x.shape(0), "A & X Dimension mismatch!")
    require(a.shape(0) == y.shape(0), "A & Y Dimension mismatch!")
    NativeBLAS.sgemv(transposeString(a), a.shape(0), a.shape(1),
      alpha, a.data, a.offset, a.shape(0),
      x.data, x.offset, 1,
      beta, y.data, y.offset, 1)
  }

  /**
   * y := alpha * x + y
   *
   * @param alpha alpha
   * @param x vector x
   * @param y vector y
   */
  def axpy(alpha: Double, x: DenseTensor[Double], y: DenseTensor[Double]): Unit = {
    require(x.size == y.size, "x and y sizes equals")
    val n = x.size
    NativeBLAS.daxpy(n, alpha, x.data, 1, y.data, 1)
  }

  /**
   * y := alpha * x + y
   *
   * @param alpha alpha
   * @param x vector x
   * @param y vector y
   */
  def axpy(alpha: Float, x: DenseTensor[Float], y: DenseTensor[Float]): Unit = {
    require(x.size == y.size, "x and y sizes equals")
    val n = x.size
    NativeBLAS.saxpy(n, alpha, x.data, 1, y.data, 1)
  }

  /**
   * x := alpha * x
   * @param alpha alpha
   * @param x vector x
   */
  def scal(alpha: Double, x: DenseTensor[Double]): Unit = {
    val n = x.size
    NativeBLAS.dscal(n, alpha, x.data, x.offset, 1)
  }

  /**
   * x := alpha * x
   * @param alpha alpha
   * @param x x
   */
  def scal(alpha: Float, x: DenseTensor[Float]): Unit = {
    val n = x.size
    NativeBLAS.sscal(n, alpha, x.data, x.offset, 1)
  }

  protected def elementwise(
  a: DenseTensor[Double],
  b: DenseTensor[Double],
  op: (Double, Double) => Double): Unit = {
    require(a.size == b.size, "Tensors of different size")
    var i = 0
    val sz = a.size
    while (i < sz) {
      a.data(i) = op(a.data(i), b.data(i))
      i += 1
    }
  }

  /**
   * Elementwise product a := a * b
   *
   * @param a vector a
   * @param b vector b
   */
  def elementwiseProduct(a: DenseTensor[Double], b: DenseTensor[Double]): Unit = {
    elementwise(a, b, (x, y) => x * y)
  }
}
