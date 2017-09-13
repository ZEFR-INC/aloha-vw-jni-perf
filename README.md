# aloha-vw-jni-perf
Aloha VW JNI performance testing

```bash
FEATURES=$(seq 1 100 | sed 's/^/f/g' | tr '\n' ' ')
CLASSES=$(seq 1 10000 | sed 's/\(.*\)/\1:-1.0 |Y _C\1_/g')

# Construct one example and feed to VW.  
# Notice the ring_size needs to be about 10 more than the number of classes.
(cat <<EOD
shared |X $FEATURES
2147483648:0.0 |y _C2147483648_
2147483649:-1 |y _C2147483649_
$CLASSES

EOD
) | $VW \
 -b 23 \
 --ring_size 10010 \
 --csoaa_ldf mc \
 --csoaa_rank \
 --loss_function logistic \
 -q YX \
 --noconstant \
 --ignore_linear X \
 --ignore y \
 --initial_weight 0.000001 \
 --invert_hash /tmp/test.1.vw.hr.model \
 -f /tmp/test.1.vw.binary.model
 ```

Model training [sample](https://github.com/eHarmony/aloha/blob/163-multilabel/aloha-vw-jni/src/test/scala/com/eharmony/aloha/models/vw/jni/multilabel/VwMultilabelModelTest.scala#L203-L216)

### Helper functions
```scala
// Returns a training line that can be passed to VW to train a model.
def vwTrainingExample(nFeatures: Int, nLabels: Int): Array[String] = {
  val features = (1 to nFeatures).map(i => s"f$i").mkString(" ")
  val labels = (1 to nLabels).map(i => s"$i:-1 |Y _${i}")
  val dummyLabels = Seq(
    "2147483648:0 |y _neg_",
    "2147483649:-1 |y _pos_"
  )

  Array(s"shared |X $features") ++ dummyLabels ++ labels
}

// Returns the VW args that should be used to train the VW model.
// Ouputs the model to `dest`.
def vwArgs(dest: java.io.File, nLabels: Int, bits: Int): String = {
  Seq(
    s"-b $bits",
    s"--ring_size ${nLabels + 10}",
    "--csoaa_ldf mc",
    "--csoaa_rank",
    "--loss_function logistic",
    "-q YX",
    "--noconstant",
    "--ignore_linear X",
    "--ignore y",
    "--initial_weight 0.000001",
    "-f " + dest.getCanonicalPath
  ).mkString(" ")
}

// Create one feature that can be used over and over.
import com.eharmony.aloha.semantics.func._
val EmptyIndicatorFn = GenFunc0(
  """Iterable(("", 1d))""", 
  (_: Any) => Iterable(("", 1d))
)

// Create feature names and feature functions that can be passed to the MultilabelModel.
// Since Vector is covariant and GenAggFunc's input is contravariant, this could be 
//
//    val (names, fns: sci.IndexedSeq[GenAggFunc[A, Iterable[(String, Double)]]]) = featureFns(10)
//
// for any type `A`
//
def featureFns(nFeatures: Int): (Vector[String], Vector[GenAggFunc[Any, Iterable[(String, Double)]]]) = {
  val names = (1 to nFeatures).map(i => s"f$i").toVector
  val fns = Vector.fill(nFeatures)(EmptyIndicatorFn)
  (names, fns)
}

// GenAggFunc is contravariant in its input and covariant in its output
// so this could be the following for any A, without casting:
// 
//   GenAggFunc[A, sci.IndexedSeq[K]]
//
def labelExtractor[K](labels: Vector[K]): GenAggFunc[Any, Vector[K]] =
  GenFunc0("[the labels]", (_: Any) => labels)
```
