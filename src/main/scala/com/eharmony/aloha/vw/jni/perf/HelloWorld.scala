package com.eharmony.aloha.vw.jni.perf

import java.io.File

import com.eharmony.aloha.audit.impl.OptionAuditor
import com.eharmony.aloha.id.ModelId
import com.eharmony.aloha.io.sources.{ExternalSource, ModelSource}
import com.eharmony.aloha.io.vfs.Vfs
import com.eharmony.aloha.models.Model
import com.eharmony.aloha.models.multilabel.MultilabelModel
import com.eharmony.aloha.models.vw.jni.multilabel.VwSparseMultilabelPredictorProducer
import com.eharmony.aloha.semantics.func.{GenAggFunc, GenFunc0}
import com.eharmony.aloha.util.SerializabilityEvidence
import org.openjdk.jmh.annotations._
import vowpalWabbit.learner.{VWActionScoresLearner, VWLearners}

import scala.collection.immutable

@State(Scope.Thread)
class HelloWorld {

//  @Benchmark
//  @BenchmarkMode(Array(Mode.Throughput))
//  @OutputTimeUnit(TimeUnit.SECONDS)
//  def measureThroughput: Unit = TimeUnit.MILLISECONDS.sleep(100)

//  @Benchmark
//  @BenchmarkMode(Array(Mode.AverageTime))
//  @OutputTimeUnit(TimeUnit.MILLISECONDS)
//  def measureAvgTime = TimeUnit.MILLISECONDS.sleep(100)

  // val model = Model

//  @Benchmark
//  def helloWorld(state: vwModel): Unit = {
//    val model = state.Model
//
//     println("hi")
//    val out: Set[(RootedTree[Any, Map[Label, Double]], Set[Label], Map[Label, Double])] = for {
//      labels <- powerSet(AllLabels.toSet).filter(ls => ls.nonEmpty)
//      x = labels.mkString(",")
//      y = model(x)
//    } yield (y, labels, ExpectedMarginalDist)
//  }

  // model.close()
  import HelloWorld._

  var x: Model[Any, Option[Map[Label, Double]]] = _

  @Setup
  def prepare: Unit = x = model()

  @Benchmark
  def helloWorld(): Unit = {
    //
  }

  @TearDown
  def tearDown: Unit = {
    // Delete the file
    model.close()
  }
}

object HelloWorld {
  def main(args: Array[String]) = {
    // model.close()
  }

  type Label = String

  def getFeatureNames(nFeatures: Int): immutable.IndexedSeq[String]  = (1 to nFeatures).map(i => s"f$i")

  def getLabelNames(nLabels: Int): immutable.IndexedSeq[String] = (1 to nLabels).map(i => s"label$i")

  // Returns a training line that can be passed to VW to train a model.
  def vwTrainingExample(nFeatures: Int, nLabels: Int): Array[String] = {
    val features = getFeatureNames(nFeatures).mkString(" ")
    val labels = (1 to nLabels).map(i => s"$i:-1 |Y _$i")
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
  val EmptyIndicatorFn = GenFunc0("""Iterable(("", 1d))""", (_: Any) => Iterable(("", 1d)))

  // Create feature names and feature functions that can be passed to the MultilabelModel.
  // Since Vector is covariant and GenAggFunc's input is contravariant, this could be
  //
  // val (names, fns: sci.IndexedSeq[GenAggFunc[A, Iterable[(String, Double)]]]) = featureFns(10)
  //
  // for any type `A`
  //
  def featureFns(nFeatures: Int): (Vector[String], Vector[GenAggFunc[Any, Iterable[(String, Double)]]]) = {
    val names = getFeatureNames(nFeatures).toVector
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

  private val nFeatures = 20
  private val nLabels = 100
  private val nLabelsDesired = 2
  private val bits = 22

  private def tmpFile() = {
    val f = File.createTempFile(classOf[HelloWorld].getSimpleName + "_", ".vw.model")
    // f.deleteOnExit()
    f
  }

  private val trainingData = vwTrainingExample(nFeatures, nLabels)

  def trainedModel(): ModelSource = {
    val modelFile = tmpFile()
    val params = vwArgs(modelFile, nLabels, bits)
    println(params)
    val learner = VWLearners.create[VWActionScoresLearner](params)

    learner.learn(trainingData)

    learner.close()

    ExternalSource(Vfs.javaFileToAloha(modelFile))
  }

  def predProd() = VwSparseMultilabelPredictorProducer[Label](
    modelSource = trainedModel,
    params      = "", // to see the output:  "-p /dev/stdout",
    defaultNs   = List.empty[Int],
    namespaces  = List(("X", List(0)))
  )

  private val Auditor = OptionAuditor[Map[Label, Double]]()

  def model() = {
    println("this is it" + new scala.Predef.ArrowAssoc(""))

    MultilabelModel(
      ModelId(1, "model"),
      getFeatureNames(nFeatures),
      featureFns(nFeatures)._2,
      getLabelNames(nLabels),
      Option(labelExtractor(getLabelNames(nLabelsDesired).toVector)),
      predProd(),
      Option.empty[Int],
      Auditor
    )(implicitly[SerializabilityEvidence[Label]])
  }


}

//object HelloWorld {
//
//  @State(Scope.Benchmark)
//  class vwModel {
//    private type Label  = String
//    private type Domain = String
//    private type PredictionOutput = RootedTree[Any, Map[Label, Double]]
//
//    private[this] val TrainingEpochs = 30
//
//    private val LabelSix   = "six"
//    private val LabelSeven = "seven"
//    private val LabelEight = "eight"
//
//    private val ExpectedMarginalDist = Map(
//      LabelSix   -> 0.6,
//      LabelSeven -> 0.7,
//      LabelEight -> 0.8
//    )
//
//    private val AllLabels = Vector(LabelSeven, LabelEight, LabelSix)
//
//    lazy val Model: Model[Domain, RootedTree[Any, Map[Label, Double]]] = {
//      val featureNames = Vector(FeatureName)
//      val features = Vector(GenFunc0("", (_: Domain) => Iterable(("", 1d))))
//
//      // Get the list of labels from the comma-separated list passed in the input string.
//      val labelsOfInterestFn =
//        GenFunc0("",
//          (x: Domain) =>
//            x.split("\\s*,\\s*")
//              .map(label => label.trim)
//              .filter(label => label.nonEmpty)
//              .toVector
//        )
//
//      val predProd = VwSparseMultilabelPredictorProducer[Label](
//        modelSource = TrainedModel,
//        params      = "", // to see the output:  "-p /dev/stdout",
//        defaultNs   = List.empty[Int],
//        namespaces  = List(("X", List(0)))
//      )
//
//      MultilabelModel(
//        modelId             = ModelId(1, "model"),
//        featureNames        = featureNames,
//        featureFunctions    = features,
//        labelsInTrainingSet = AllLabels,
//        labelsOfInterest    = Option(labelsOfInterestFn),
//        predictorProducer   = predProd,
//        numMissingThreshold = Option.empty[Int],
//        auditor             = Auditor)
//    }
//
//    private def tmpFile() = {
//      val f = File.createTempFile(classOf[HelloWorld].getSimpleName + "_", ".vw.model")
//      f.deleteOnExit()
//      f
//    }
//
//    private def vwTrainingParams(modelFile: File = tmpFile()) = {
//
//      // NOTES:
//      //  1. `--csoaa_rank`  is needed by VW to make a VWActionScoresLearner.
//      //  2. `-q YX`  forms the cross product of features between the label-based features in Y
//      //     and the side information in X.  If the features in namespace Y are unique to each
//      //     class, the cross product effectively makes one model per class by interacting the
//      //     class features to the features in X.  Since the class labels vary (possibly)
//      //     independently, this cross product is capable of creating independent models
//      //     (one per class).
//      //  3. `--ignore_linear Y`  is provided because in the data, there is one constant feature
//      //     in the Y namespace per class.  This flag acts essentially the same as the
//      //     `--noconstant`  flag in the traditional binary classification context.  It omits the
//      //     one feature related to the class (the intercept).
//      //  4. `--noconstant`  is specified because there's no need to have a "common intercept"
//      //     whose value is the same across all models.  This should be the job of the
//      //     first-order features in Y (the per-class intercepts).
//      //  5. `--ignore y`  is used to ignore the features in the namespace related to the dummy
//      //     classes.  We need these two dummy classes in training to make  `--csoaa_ldf mc`
//      //     output proper probabilities.
//      //  6. `--link logistic`  doesn't actually seem to do anything.
//      //  7. `--loss_function logistic`  works properly during training; however, when
//      //     interrogating the scores, it is important to remember they appear as
//      //     '''negative logits'''.  This is because the CSOAA algorithm uses '''costs''', so
//      //     "smaller is better".  So, to get the probability, one must do `1/(1 + exp(-1 * -y))`
//      //     or simply `1/(1 + exp(y))`.
//      val flags =
//      """
//        | --quiet
//        | --csoaa_ldf mc
//        | --csoaa_rank
//        | --loss_function logistic
//        | -q YX
//        | --noconstant
//        | --ignore_linear X
//        | --ignore y
//        | -f
//      """.stripMargin.trim
//
//      (flags + " " + modelFile.getCanonicalPath).split("\n").map(_.trim).mkString(" ")
//    }
//
//    private val FeatureName = "feature"
//
//    /**
//      * A dataset that creates the following marginal distribution.
//    - Pr[seven] = 0.7  where seven is _C0_
//    - Pr[eight] = 0.8  where eight is _C1_
//    - Pr[six]   = 0.6  where six   is _C2_
//      *
//      * The observant reader may notice these are oddly ordered.  On each line C1 appears first,
//      * then C0, then C2.  This is done to show ordering doesn't matter.  What matters is the
//      * class '''indices'''.
//      */
//    private val TrainingData =
//      Vector(
//        s"shared |X $FeatureName\n2147483648:0.0 |y _C2147483648_\n2147483649:-0.084 |y _C2147483649_\n1:0.0 |Y _C1_\n0:-0.084 |Y _C0_\n2:-0.084 |Y _C2_",
//        s"shared |X $FeatureName\n2147483648:0.0 |y _C2147483648_\n2147483649:-0.024 |y _C2147483649_\n1:0.0 |Y _C1_\n0:0.0 |Y _C0_\n2:0.0 |Y _C2_",
//        s"shared |X $FeatureName\n2147483648:0.0 |y _C2147483648_\n2147483649:-0.336 |y _C2147483649_\n1:-0.336 |Y _C1_\n0:-0.336 |Y _C0_\n2:-0.336 |Y _C2_",
//        s"shared |X $FeatureName\n2147483648:0.0 |y _C2147483648_\n2147483649:-0.056 |y _C2147483649_\n1:0.0 |Y _C1_\n0:-0.056 |Y _C0_\n2:0.0 |Y _C2_",
//        s"shared |X $FeatureName\n2147483648:0.0 |y _C2147483648_\n2147483649:-0.144 |y _C2147483649_\n1:-0.144 |Y _C1_\n0:0.0 |Y _C0_\n2:-0.144 |Y _C2_",
//        s"shared |X $FeatureName\n2147483648:0.0 |y _C2147483648_\n2147483649:-0.224 |y _C2147483649_\n1:-0.224 |Y _C1_\n0:-0.224 |Y _C0_\n2:0.0 |Y _C2_",
//        s"shared |X $FeatureName\n2147483648:0.0 |y _C2147483648_\n2147483649:-0.036 |y _C2147483649_\n1:0.0 |Y _C1_\n0:0.0 |Y _C0_\n2:-0.036 |Y _C2_",
//        s"shared |X $FeatureName\n2147483648:0.0 |y _C2147483648_\n2147483649:-0.096 |y _C2147483649_\n1:-0.096 |Y _C1_\n0:0.0 |Y _C0_\n2:0.0 |Y _C2_"
//      ).map(_.split(raw"\n"))
//
//    private lazy val TrainedModel: ModelSource = {
//      val modelFile = tmpFile()
//      val params = vwTrainingParams(modelFile)
//      val learner = VWLearners.create[VWActionScoresLearner](params)
//
//      for {
//        _ <- 1 to TrainingEpochs
//        d <- TrainingData
//      } learner.learn(d)
//
//      learner.close()
//
//      ExternalSource(Vfs.javaFileToAloha(modelFile))
//    }
//
//    private val Auditor = RootedTreeAuditor.noUpperBound[Map[Label, Double]]()
//
//    /**
//      * Creates the power set of the provided set.
//      * Answer provided by Chris Marshall (@oxbow_lakes) on
//      * [[https://stackoverflow.com/a/11581323 Stack Overflow]].
//      * @param generatorSet a set for which a power set should be produced.
//      * @tparam A type of elements in the set.
//      * @return the power set of `generatorSet`.
//      */
//    private def powerSet[A](generatorSet: Set[A]): Set[Set[A]] = {
//      @tailrec def pwr(t: Set[A], ps: Set[Set[A]]): Set[Set[A]] =
//        if (t.isEmpty) ps
//        else pwr(t.tail, ps ++ (ps map (_ + t.head)))
//
//      // powerset of empty set is the set of the empty set.
//      pwr(generatorSet, Set(Set.empty[A]))
//    }
//  }
//
//
//}
