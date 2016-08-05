package org.apache.spark.ml.util

import org.apache.log4j.{Logger, Level}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.scalatest.{Suite, BeforeAndAfterAll}

trait SparkTestContext extends BeforeAndAfterAll { self: Suite =>
  @transient var spark: SparkSession = _
  @transient var sc: SparkContext = _
  @transient var checkpointDir: String = _

  override def beforeAll() {
    super.beforeAll()
    spark = SparkSession.builder
      .master("local[2]")
      .appName("MLlibUnitTest")
      .config("spark.sql.warehouse.dir", "xxx")
      .getOrCreate()
    sc = spark.sparkContext
    Logger.getLogger("org").setLevel(Level.WARN)
  }

  override def afterAll() {
    try {
      SparkSession.clearActiveSession()
      if (spark != null) {
        spark.stop()
      }
      spark = null
    } finally {
      super.afterAll()
    }
  }
}

