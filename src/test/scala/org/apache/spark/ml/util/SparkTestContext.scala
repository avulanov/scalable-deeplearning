package org.apache.spark.ml.util

import org.apache.log4j.{LogManager, Logger, Level}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{Suite, BeforeAndAfterAll}

trait SparkTestContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("SparkAlgorithmsUnitTest")
      //.set("spark.eventLog.enabled", "true")
      //.set("spark.eventLog.dir","log")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    sc = new SparkContext(conf)
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }
}


