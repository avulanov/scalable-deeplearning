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

import org.scalatest.FunSuite

class DenseTensorSuite  extends FunSuite {

  test ("value") {
    val data = Array[Double](1, 2, 3, 4, 5, 6, 7, 8)
    val shape2d = Array(4, 2)
    val tensor2d = DenseTensor[Double](data, shape2d)
    assert(tensor2d.value(Array(2, 1)) == 7.0, "(1, 1) must be 7.0")
    val shape3d = Array(2, 2, 2)
    val tensor3d = DenseTensor[Double](data, shape3d)
    assert(tensor3d.value(Array(1, 1, 1)) == 8.0, "(1, 1, 1) must be 8.0")
  }

  test ("slice") {
    val data8 = Array[Double](0, 1, 2, 3, 4, 5, 6, 7)
    val shape2d = Array(4, 2)
    val tensor2d = DenseTensor[Double](data8, shape2d)
    val slice2d = tensor2d.slice(1, 2)
    assert(slice2d.copyData().deep == data8.slice(4, 8).deep,
      "The resulting slice must be (4, 5, 6, 7) ")
    val data12 = Array[Double](0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    val shape3d = Array(2, 2, 3)
    val tensor3d = DenseTensor[Double](data12, shape3d)
    val slice3d = tensor3d.slice(1, 2)
    assert(slice3d.copyData().deep == data12.slice(4, 8).deep,
      "The resulting slice must be (4, 5, 6, 7) ")
    val shape5d = Array(2, 1, 2, 1, 3)
    val tensor5d = DenseTensor[Double](data12, shape5d)
    val slice5dto2d = tensor5d.slice(1)
    assert(slice5dto2d.copyData().deep == data12.slice(4, 8).deep,
      "The resulting slice must be (4, 5, 6, 7) ")
  }

  test ("apply function") {
    val shape2d = Array(4, 2)
    val a = DenseTensor[Double](Array[Double](0, 1, 2, 3, 4, 5, 6, 7), shape2d)
    DenseTensor.applyFunction(a, (t: Double) => t * t)
    assert(a.copyData().deep == Array[Double](0, 1, 4, 9, 16, 25, 36, 49).deep,
      "The result must be (1, 2, 3, 4, 5, 6, 7, 8)")
    val x = DenseTensor[Double](Array[Double](0, 1, 2, 3, 4, 5, 6, 7), shape2d)
    val y = DenseTensor[Double](shape2d)
    def func: (Double) => Double = v => v + 1
    DenseTensor.applyFunction[Double](x, y, func)
    assert(y.copyData().deep == Array[Double](1, 2, 3, 4, 5, 6, 7, 8).deep,
      "The result must be (1, 2, 3, 4, 5, 6, 7, 8)")
    val x2 = DenseTensor[Double](Array[Double](0, 1, 2, 3, 4, 5, 6, 7), shape2d)
    val x1 = x
    val z = DenseTensor[Double](shape2d)
    def func2: (Double, Double) => Double = (v1, v2) => v1 + v2
    DenseTensor.applyFunction[Double](x1, x2, z, func2)
    assert(z.copyData().deep == Array[Double](0, 2, 4, 6, 8, 10, 12, 14).deep,
      "The result must be (0, 2, 4, 6, 8, 10, 12, 14)")
  }

  test ("fillWith") {
    val recipient = DenseTensor[Double](Array(4, 2))
    val donor = DenseTensor[Double](Array[Double](0, 1, 2, 3), Array(4, 1))
    recipient.fillWith(donor)
    assert(recipient.copyData().deep == Array[Double](0, 1, 2, 3, 0, 1, 2, 3).deep,
      "The result must be (0, 1, 2, 3, 0, 1, 2, 3)")
  }

  test ("fill") {
    val onesTensor = DenseTensor.fill[Double](Array(1, 2, 1))(1.0)
    assert(onesTensor.copyData().forall(x => x == 1.0), "All elements are 1.0")
  }

  test ("plus double") {
    val x = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    val y = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    val z = x + y
    val trueZ = DenseTensor[Double](Array[Double](2, 4, 6, 8, 10, 12), Array(2, 3))
    assert(z.isEqual(trueZ), "Transposed, shape or data differs")
  }

  test ("plus float") {
    val x = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    val y = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    val z = x + y
    val trueZ = DenseTensor[Float](Array[Float](2, 4, 6, 8, 10, 12), Array(2, 3))
    assert(z.isEqual(trueZ), "Transposed, shape or data differs")
  }

  test ("minus double") {
    val x = DenseTensor[Double](Array[Double](2, 4, 6, 8, 10, 12), Array(2, 3))
    val y = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    val z = x - y
    val trueZ = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    assert(z.isEqual(trueZ), "Transposed, shape or data differs")
  }

  test ("minus float") {
    val x = DenseTensor[Float](Array[Float](2, 4, 6, 8, 10, 12), Array(2, 3))
    val y = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    val z = x - y
    val trueZ = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    assert(z.isEqual(trueZ), "Transposed, shape or data differs")
  }

  test ("elementwise product double") {
    val x = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    val y = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    val z = x :* y
    val trueZ = DenseTensor[Double](Array[Double](1, 4, 9, 16, 25, 36), Array(2, 3))
    assert(z.isEqual(trueZ), "Transposed, shape or data differs")
  }

  test ("elementwise product float") {
    val x = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    val y = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    val z = x :* y
    val trueZ = DenseTensor[Float](Array[Float](1, 4, 9, 16, 25, 36), Array(2, 3))
    assert(z.isEqual(trueZ), "Transposed, shape or data differs")
  }

  test ("sum double") {
    val x = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    assert(x.sum == 21, "Sum has to be 21")
  }

  test ("sum float") {
    val x = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    assert(x.sum == 21, "Sum has to be 21")
  }

  test ("axpy double precision") {
    val alpha = 2
    val x = DenseTensor[Double](Array[Double](0.5, 1, 1.5, 2, 2.5, 3), Array(6))
    val y = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(6))
    DenseTensor.axpy(alpha, x, y)
    assert(y.copyData().deep == Array[Double](2, 4, 6, 8, 10, 12).deep)
  }

  test ("axpy single precision") {
    val alpha = 2
    val x = DenseTensor[Float](Array[Float](0.5f, 1f, 1.5f, 2f, 2.5f, 3f), Array(6))
    val y = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(6))
    DenseTensor.axpy(alpha, x, y)
    assert(y.copyData().deep == Array[Float](2, 4, 6, 8, 10, 12).deep)
  }

  test ("dgemm double precision") {
    val a = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    val b = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(3, 2))
    val c = DenseTensor[Double](Array(2, 2))
    DenseTensor.gemm(1.0, a, b, 0.0, c)
    assert(c.copyData().deep == Array[Double](22, 28, 49, 64).deep)
    DenseTensor.gemm(0.5, a, b, 0.5, c)
    assert(c.copyData().deep == Array[Double](22, 28, 49, 64).deep)
  }

  test ("dgemm double precision transpose") {
    val a = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(3, 2))
    val b = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(3, 2))
    val c = DenseTensor[Double](Array(2, 2))
    DenseTensor.gemm(1.0, a.transpose, b, 0.0, c)
    assert(c.copyData().deep == Array[Double](14, 32, 32, 77).deep)
  }

  test ("dgemm single precision") {
    val a = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    val b = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(3, 2))
    val c = DenseTensor[Float](Array(2, 2))
    DenseTensor.gemm(1.0f, a, b, 0.0f, c)
    assert(c.copyData().deep == Array[Float](22, 28, 49, 64).deep)
    DenseTensor.gemm(0.5f, a, b, 0.5f, c)
    assert(c.copyData().deep == Array[Float](22, 28, 49, 64).deep)
  }

  test ("dgemm single precision transpose") {
    val a = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(3, 2))
    val b = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(3, 2))
    val c = DenseTensor[Float](Array(2, 2))
    DenseTensor.gemm(1.0f, a.transpose, b, 0.0f, c)
    assert(c.copyData().deep == Array[Double](14, 32, 32, 77).deep)
  }

  test("gemv double precision") {
    val a = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    val x = DenseTensor[Double](Array[Double](1, 2, 3), Array(3))
    val y = DenseTensor[Double](Array[Double](2, 2), Array(2))
    DenseTensor.gemv(1.0, a, x, 0.5, y)
    assert(y.copyData().deep == Array[Double](23, 29).deep)
  }

  test("gemv single precision") {
    val a = DenseTensor[Float](Array[Float](1, 2, 3, 4, 5, 6), Array(2, 3))
    val x = DenseTensor[Float](Array[Float](1, 2, 3), Array(3))
    val y = DenseTensor[Float](Array[Float](2, 2), Array(2))
    DenseTensor.gemv(1.0f, a, x, 0.5f, y)
    assert(y.copyData().deep == Array[Float](23, 29).deep)
  }

  test ("elementwise product") {
    val a = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    val b = DenseTensor[Double](Array[Double](1, 2, 3, 4, 5, 6), Array(2, 3))
    DenseTensor.elementwiseProduct(a, b)
    assert(a.copyData().deep == Array[Double](1, 4, 9, 16, 25, 36).deep)
  }
}
