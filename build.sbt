name := "scalable-deeplearning"

version := "1.0.0"

scalaVersion := "2.11.7"

spName := "avulanov/scalable-deeplearning"

spShade := true

sparkVersion := "2.0.0"

libraryDependencies ++= Seq(
  "com.github.fommil.netlib" % "all" % "1.1.2",
  "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test"
)

sparkComponents += "mllib"

// libraryDependencies ++= Seq(
//  "org.apache.spark" % "spark-core_2.11" % "2.0.0" % "provided",
//  "org.apache.spark" % "spark-mllib_2.11" % "2.0.0" % "provided"
// )

test in assembly := {}

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)