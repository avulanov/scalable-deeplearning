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

import scaladl.tensor.Math.NumberLike

object Math {
  trait NumberLike[@specialized (Double, Int) T] {
    def plus(x: T, y: T): T
  }
  object NumberLike {
    implicit object NumberLikeDouble extends NumberLike[Double] {
      def plus(x: Double, y: Double): Double = x + y
    }
    implicit object NumberLikeInt extends NumberLike[Int] {
      def plus(x: Int, y: Int): Int = x + y
    }
  }
}
object Statistics {
  import Math.NumberLike
  def plus[@specialized (Double, Int) T](x: T, y: T)(implicit ev: NumberLike[T]): T =
    ev.plus(x, y)
  def plusDouble(x: Double, y: Double): Double = x + y
}

class My[@specialized (Double, Int) T](implicit ev: NumberLike[T]) {
  def plus(x: T, y: T): T = ev.plus(x, y)
}

object TypedClassTest {
  def main(args: Array[String]): Unit = {
//    Statistics.plus(2.0, 2.0)
//    Statistics.plusDouble(2.0, 2.0)
    val m = new My[Double]()
    m.plus(2.0, 2.0)
  }
}
