# A Scalable Implementation of Deep Learning on Spark
This library is based on the implementation of artificial neural networks in [Spark ML](https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier). In addition to the multilayer perceptron, it contains new [Spark deep learning features](https://issues.apache.org/jira/browse/SPARK-5575) that were not yet merged to Spark ML. Currently, they are Stacked Autoencoder and tensor data flow. Highlights of the library:
  - Provides Spark ML pipeline API
  - Implements data parallel training
  - Supports native CPU BLAS
  - Employs tensor data flow
  - Provides extensible API for developers of new features

## Installation
### Requirements
  - Apache Spark 2.0 or higher
  - Java and Scala
  - Maven

### Build 
Clone and compile:
```
git clone https://github.com/avulanov/scalable-deeplearning.git
cd scalable-deeplearning
mvn package
```
The jar library will be availabe in `target` folder.

### Performance configuration
For the best performance, native BLAS library should be in the path of all nodes that run Spark. The required name is `libblas.so.3`. OpenBLAS is recommended. Below are the setup details for different platforms. Indication of successfull use of BLAS is the following line in Spark logs:
```
INFO JniLoader: successfully loaded ...netlib-native_system-....dll
```
### Linux:
Install native blas library (depending on your distributive):
```
yum install openblas <OR> apt-get openblas <OR> download and compile OpenBLAS
```
Create symlink to native BLAS within its folder `/your/blas`
```
ln -s libopenblas.so libblas.so.3
```
Add it to your library path. Make sure there is no other folder with `libblas.so.3` in your path.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/blas
```
### Windows:
Copy the following dlls from MINGW distribution and from OpenBLAS to the folder `blas`. Make sure they are all the same 64 or 32 bit. Add that folder to your `path` variable.
```
libquadmath-0.dll // MINGW
libgcc_s_seh-1.dll // MINGW
libgfortran-3.dll // MINGW
libopeblas.dll // OpenBLAS binary
liblapack3.dll // copy of libopeblas.dll
libblas3.dll // copy of libopenblas.dll
```
  - MinGW https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Automated%20Builds/
  - OpenBLAS http://www.openblas.net/

## Example of use
Start Spark with this library:
```
./spark-shell --jars scaladl.jar
```
Or use it as external dependency for your application.

### Multilayer perceptron
```scala
import org.apache.spark.ml.scaladl.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame.
// MNIST handwritten recognition data 
// https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
val train = spark.read.format("libsvm").option("numFeatures", 784).load("mnist.scale")
val test =  spark.read.format("libsvm").option("numFeatures", 784).load("mnist.scale.t")
// specify layers for the neural network:
// input layer of size 784 (features), one hidden layer of size 100
// and output of size 10 (classes)
val layers = Array[Int](784, 32, 10)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)
// train the model
val model = trainer.fit(train)
// compute accuracy on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")
println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
```
On a single machine after ~2 minutes:
```
Accuracy: 0.9616
```
### Stacked Autoencoder
```scala
import org.apache.spark.ml.scaladl.StackedAutoencoder
// Load the data stored in LIBSVM format as a DataFrame.
// MNIST handwritten recognition data 
// https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
val train = spark.read.format("libsvm").option("numFeatures", 784).load("mnist.scale")
// create autoencoder and decode with one hiddel layer of 32 neurons
val stackedAutoencoder = new StackedAutoencoder()
  .setLayers(Array(784, 32))
  .setBlockSize(1)
  .setMaxIter(100)
  .setSeed(123456789L)
  .setTol(1e-6)
  .setInputCol("input")
  .setOutputCol("output")
  .setDataIn01Interval(is01)
  .setBuildDecoder(true)
val saModel = stackedAutoencoder.fit(df)
saModel.setInputCol("input").setOutputCol("encoded")
// encoding
val encodedData = saModel.transform(df)
// decoding
saModel.setInputCol("encoded").setOutputCol("decoded")
val decodedData = saModel.decode(encodedData)
```
