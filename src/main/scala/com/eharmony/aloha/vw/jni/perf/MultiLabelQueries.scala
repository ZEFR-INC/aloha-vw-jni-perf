package com.eharmony.aloha.vw.jni.perf

import java.io.File

import com.eharmony.aloha.audit.impl.OptionAuditor
import com.eharmony.aloha.dataset.density.Sparse
import com.eharmony.aloha.id.ModelId
import com.eharmony.aloha.io.sources.{ExternalSource, ModelSource}
import com.eharmony.aloha.io.vfs.Vfs
import com.eharmony.aloha.models.Model
import com.eharmony.aloha.models.multilabel.MultilabelModel
import com.eharmony.aloha.models.vw.jni.multilabel.VwSparseMultilabelPredictorProducer
import com.eharmony.aloha.semantics.func.{GenAggFunc, GenFunc0}
import org.openjdk.jmh.annotations._
import vowpalWabbit.learner.{VWActionScoresLearner, VWLearners}
import vowpalWabbit.responses.ActionScores

import scala.collection.immutable
import scala.util.Random

@State(Scope.Thread)
class MultiLabelQueries {
  import MultiLabelQueries._

  @Param(Array("200", "500", "1000"))
  private var nLabels: Int = _

  @Param(Array("0.01", "0.1", "0.5", "1"))
  private var nLabelsQueriedPercentage: Double = _

  @Param(Array("10", "50", "100"))
  private var nFeatures: Int = _

  @Param(Array("20", "22"))
  private var bits: Int = _

  @Param(Array("1e-5", "1e-6"))
  private var initialWeights: Long = _

  private var model: Model[Domain, Option[Map[Label, Double]]] = _
  private var vwModel: VWActionScoresLearner = _
  private var vwTestExample: Array[String] = _
  private var alohaTestExample = _

  @Setup
  def prepare(): Unit = {
    val trainingData: Array[String] = vwTrainingExample(nFeatures, nLabels)
    val trainedModel: ModelSource = getTrainedModel(nLabels, bits, trainingData, initialWeights)
    val vwModelParams = "-t --quiet -i " + trainedModel.localVfs.descriptor
    val nLabelsQueried = math.round(nLabelsQueriedPercentage * nLabels).toInt
    alohaTestExample = null
    model = getModel(trainedModel, nFeatures, nLabels, nLabelsQueried)
    vwModel = VWLearners.create[VWActionScoresLearner](vwModelParams)

    // Keep the shared features and the first nLabelsQueried
    // The format is:
    // shared features on first line, 2 dummy labels on the next two lines (so we can drop 3)
    // and then the labels
    vwTestExample = trainingData(0) +: trainingData.slice(3, nLabelsQueried + 3)
  }

  @Benchmark
  def aloha(): Option[Map[Label, Double]] = model(alohaTestExample)

  @Benchmark
  def vw(): ActionScores = vwModel.predict(vwTestExample)

  @TearDown
  def tearDown(): Unit = {
    // The file is not deleted because (we think) it is automatically deleted on JVM exit
    model.close()
    vwModel.close()
  }
}

object MultiLabelQueries {
  type Domain = Any
  type Label = Int

  def getFeatureNames(nFeatures: Int): immutable.IndexedSeq[String]  = (1 to nFeatures).map(i => s"f$i")

  def getLabels(nLabels: Int): Vector[Label] = (1 to nLabels).toVector

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
  def vwArgs(dest: java.io.File, nLabels: Int, bits: Int, initialWeight: Double): String = {
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
      s"--initial_weight $initialWeight",
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
  private def featureFns(
    nFeatures: Int
  ): (Vector[String], Vector[GenAggFunc[Any, Iterable[(String, Double)]]]) = {
    val names = getFeatureNames(nFeatures).toVector
    val fns = Vector.fill(nFeatures)(EmptyIndicatorFn)
    (names, fns)
  }

  // GenAggFunc is contravariant in its input and covariant in its output
  // so this could be the following for any A, without casting:
  //
  //   GenAggFunc[A, sci.IndexedSeq[K]]
  //
  private def labelExtractor[K](labels: Vector[K]): GenAggFunc[Any, Vector[K]] =
  GenFunc0("[the labels]", (_: Any) => labels)

  private def tmpFile(): File = {
    val f = File.createTempFile(classOf[MultiLabelQueries].getSimpleName + "_", ".vw.model")
    f.deleteOnExit()
    f
  }

  private def getTrainedModel(nLabels: Int, bits: Int, trainingData: Array[String],
    initialWeight: Double):
  ModelSource = {
    val modelFile = tmpFile()
    val params = vwArgs(modelFile, nLabels, bits, initialWeight)
    val learner = VWLearners.create[VWActionScoresLearner](params)
    learner.learn(trainingData)
    learner.close()
    ExternalSource(Vfs.javaFileToAloha(modelFile))
  }

  private def predProd(trainedModel: ModelSource): VwSparseMultilabelPredictorProducer[Label] =
    VwSparseMultilabelPredictorProducer[Label](
    modelSource = trainedModel,
    params      = "", // to see the output:  "-p /dev/stdout",
    defaultNs   = List.empty[Int],
    namespaces  = List(("X", List(0)))
  )

  private val Auditor: OptionAuditor[Map[Label, Double]] = OptionAuditor[Map[Label, Double]]()

  def createInitialWeight(seed: Long): Double = {
    val random = new Random(seed)
    val sign = if (random.nextFloat() > 0.5) 1 else -1
    1e-6 + sign * random.nextFloat() * 1e-7
  }

  def getModel(
    trainedModel: ModelSource,
    nFeatures: Int,
    nLabels: Int,
    nLabelsQueried: Int
  ): Model[Domain, Option[Map[Label, Double]]] = {
    val (featureNames, features) = featureFns(nFeatures)
    val featureFunctions: Vector[GenAggFunc[Domain, Sparse]] = features

    MultilabelModel(
      ModelId(1, "model"),
      featureNames,
      featureFunctions,
      getLabels(nLabels),
      Option(labelExtractor(getLabels(nLabelsQueried))),
      predProd(trainedModel),
      Option.empty[Int],
      Auditor
    )
  }
}
