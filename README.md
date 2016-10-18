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
sbt assembly (or mvn assembly)
```
The jar library will be availabe in `target` folder. `assembly` includes optimized numerical processing library netlib-java. Optionally, one can build `package`.

### Performance configuration
Scaladl uses [netlib-java](https://github.com/fommil/netlib-java) library for optimized numerical processing with native [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). All netlib-java classes are included in scaladl.jar. The latter has to be in the classpath before Spark's own libraries because Spark has a subset of netlib. In order to do this, set `spark.driver.userClassPathFirst` to `true` in `spark-defaults.conf`.

If native BLAS libraries are not available at runtime or scaladl is not the first in the classpath, you will see a warning `WARN BLAS: Failed to load implementation from:` and reference or pure JVM implementation will be used. Native BLAS library such as OpenBLAS (`libopenblas.so` or `dll`) or ATLAS (`libatlas.so`) should be in the path of all nodes that run Spark. Netlib-java requires the library to be named as `libblas.so.3`, and one has to create a symlink. The same is for Windows and `libblas3.dll`. Below are the setup details for different platforms. With proper configuration, you will see an info `INFO JniLoader: successfully loaded ...netlib-native_system-....`

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
### Built-in examples
Scaldl provides working examples of MNIST classification and pre-training with stacked autoencoder. Examples are in [`scaladl.examples`](https://github.com/avulanov/scalable-deeplearning/tree/master/src/main/scala/scaladl/examples) package. They can be run via Spark submit:
```
./spark-submit --class scaladl.examples.MnistClassification --master spark://master:7077 /path/to/scalable-deeplearning-assembly-1.0.0.jar /path/to/mnist-libsvm
```
### Spark shell
Start Spark with this library:
```
./spark-shell --jars /path/to/scalable-deeplearning-assembly-1.0.0.jar
```
Or use it as external dependency for your application.

### Multilayer perceptron
MNIST classification
  - Load MNIST handwritten recognition data stored in [LIBSVM format](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html) as a DataFrame
  - Initialize the multilayer perceptron classifier with 784 inputs, 32 neurons in hidden layer and 10 outputs
  - Train and predict

```scala
import org.apache.spark.ml.scaladl.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val train = spark.read.format("libsvm").option("numFeatures", 784).load("mnist.scale").persist()
val test = spark.read.format("libsvm").option("numFeatures", 784).load("mnist.scale.t").persist()
train.count() // materialize data lazy persisted in memory
test.count() // materialize data lazy persisted in memory
val trainer = new MultilayerPerceptronClassifier().setLayers(Array(784, 32, 10)).setMaxIter(100)
val model = trainer.fit(train)
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")
println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
```
### Stacked Autoencoder
Pre-training
  - Load MNIST data
  - Initialize the stacked autoencoder with 784 inputs and 32 neurons in hidden layer
  - Train stacked autoencoder
  - Initialize the multilayer perceptron classifier with 784 inputs, 32 neurons in hidden layer and 
```scala
import org.apache.spark.ml.scaladl.{MultilayerPerceptronClassifier, StackedAutoencoder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val train = spark.read.format("libsvm").option("numFeatures", 784).load("mnist.scale").persist()
val test = spark.read.format("libsvm").option("numFeatures", 784).load("mnist.scale.t").persist()
train.count()
test.count()
val stackedAutoencoder = new StackedAutoencoder().setLayers(Array(784, 32))
  .setInputCol("features")
  .setOutputCol("output")
  .setDataIn01Interval(true)
  .setBuildDecoder(false)
val saModel = stackedAutoencoder.fit(train)
val autoWeights = saModel.encoderWeights
val trainer = new MultilayerPerceptronClassifier().setLayers(Array(784, 32, 10)).setMaxIter(1)
val initialWeights = trainer.fit(train).weights
System.arraycopy(autoWeights.toArray, 0, initialWeights.toArray, 0, autoWeights.toArray.length)
trainer.setInitialWeights(initialWeights).setMaxIter(10)
val model = trainer.fit(train)
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")
println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
```
## Contributions
Contributions are welcome, in particular in the following areas:
  - New layers
    - Convolutional
    - ReLu
  - Flexibility
    - Implement the reader of Caffe/other deep learning configuration format
    - Implement Python/R/Java interface
  - Efficiency
    - Switch from double to single precision 
    - Implement wrapper to specialized deep learning libraries, e.g. TensorFlow
  - Refactoring
    - Implement own version of L-BFGS to remove dependency on breeze
