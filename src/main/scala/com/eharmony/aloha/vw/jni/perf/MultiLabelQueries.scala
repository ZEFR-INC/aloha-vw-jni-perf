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

/**
  * Test the model performance of Aloha's MultilabelModel and the underlying VW JNI model.
  * @author amir.ziai, ryan.deak
  */
@State(Scope.Thread)
class MultiLabelQueries {
  import MultiLabelQueries._

  // =========================   JMH TEST PARAMS   =========================

  @Param(Array("200", "500", "1000"))
  private var nLabels: Int = _

  @Param(Array("0.01", "0.1", "0.5", "1"))
  private var nLabelsQueriedPercentage: Double = _

  @Param(Array("10", "50", "100"))
  private var nFeatures: Int = _

  @Param(Array("20", "22"))
  private var bits: Int = _

  @Param(Array("1e-5", "1e-6"))
  private var initialWeights: Double = _


  // =========================   STATE VARIABLES   =========================

  private var model: Model[Domain, Option[Map[Label, Double]]] = _
  private var vwModel: VWActionScoresLearner = _
  private var vwTestExample: Array[String] = _
  private var alohaTestExample: Any = _

  @Setup
  def prepare(): Unit = {
    val trainingData: Array[String] = vwTrainingExample(nFeatures, nLabels)
    val trainedModel: ModelSource = getTrainedModel(nLabels, bits, trainingData, initialWeights)
    val nLabelsQueried = math.round(nLabelsQueriedPercentage * nLabels).toInt
    val vwModelParams = s"--ring_size ${nLabels + 5} -t --quiet -i " + trainedModel.localVfs.descriptor

    // Update the state.

    alohaTestExample = null
    model = getModel(trainedModel, nFeatures, nLabels, nLabelsQueried)
    vwModel = VWLearners.create[VWActionScoresLearner](vwModelParams)

    // Dummy labels are not necessary for prediction.
    // Keep the shared features and the labels, but remove the dummy labels.
    vwTestExample = trainingData(0) +: trainingData.slice(3, nLabelsQueried + 3)
  }

  @TearDown
  def tearDown(): Unit = {
    // The file is not deleted manually because (we think) it is automatically deleted on JVM exit
    model.close()
    vwModel.close()
  }


  // ============================   BENCHMARKS   ===========================

  /**
    * Use Aloha to predict one example.
    * @return Aloha prediction output.
    */
  @Benchmark
  def aloha(): Option[Map[Label, Double]] = model(alohaTestExample)

  /**
    * Use VW JNI to predict one example.
    * @return VW JNI prediction output.
    */
  @Benchmark
  def vw(): ActionScores = vwModel.predict(vwTestExample)
}

object MultiLabelQueries {

  /**
    * The model doesn't need to use the input because the label extractor returns a constant
    * list of labels once the model is constructed.
    */
  private type Domain = Any

  /**
    * Use `Int` labels to make them as simple as possible.  We don't care about the actual
    * label values.
    */
  private type Label = Int

  /**
    * Change this to show VW output while training the model.
    */
  private[this] val ShowVwDiagnostics: Boolean = false

  private[this] def getFeatureNames(nFeatures: Int): immutable.IndexedSeq[String] =
    (1 to nFeatures).map(i => s"f$i")

  private[this] def getLabels(nLabels: Int): Vector[Label] = (1 to nLabels).toVector

  // Returns the VW args that should be used to train the VW model.
  // Ouputs the model to `dest`.
  private[this] def vwArgs(dest: File, nLabels: Int, bits: Int, initialWeight: Double): String = {
    Seq(
      if (ShowVwDiagnostics) "" else "--quiet",
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
    ).mkString(" ").trim
  }

  // Create one feature that can be used over and over.
  private[this] val EmptyIndicatorFn =
    GenFunc0("""Iterable(("", 1d))""", (_: Any) => Iterable(("", 1d)))

  // Create feature names and feature functions that can be passed to the MultilabelModel.
  // Since Vector is covariant and GenAggFunc's input is contravariant, this could be
  //
  // val (names, fns: sci.IndexedSeq[GenAggFunc[A, Iterable[(String, Double)]]]) = featureFns(10)
  //
  // for any type `A`
  //
  private[this] def featureFns(
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
  private[this] def labelExtractor[K](labels: Vector[K]): GenAggFunc[Any, Vector[K]] =
    GenFunc0("[the labels]", (_: Any) => labels)

  private[this] def tmpFile(): File = {
    val f = File.createTempFile(classOf[MultiLabelQueries].getSimpleName + "_", ".vw.model")
    f.deleteOnExit()
    f
  }

  private[this] def predictorProducer(
      trainedModel: ModelSource,
      nLabels: Int
  ): VwSparseMultilabelPredictorProducer[Label] =
    VwSparseMultilabelPredictorProducer[Label](
    modelSource = trainedModel,
    params      = s"--ring_size ${nLabels + 10}", // to see the output:  "-p /dev/stdout",
    defaultNs   = List.empty[Int],
    namespaces  = List(("X", List(0)))
  )

  /**
    * Use an OptionAuditor because it offers highest performance.  It returns the least amount of
    * extraneous information.
    */
  private[this] val Auditor: OptionAuditor[Map[Label, Double]] = OptionAuditor[Map[Label, Double]]()


  /**
    * The training data format is:
    *
    - ''index 0'': shared features
    - ''index 1'': negative dummy example (necessary to normalize probs correctly)
    - ''index 2'': positive dummy example (necessary to normalize probs correctly)
    - ''indices 3 ... nLabels + 3 (exclusive)'': label-based features.
    *
    * @param nFeatures number of shared features to add to the example.
    * @param nLabels number of labels to add to the example.
    * @return a training line that can be passed to VW to train a model.
    */
  private def vwTrainingExample(nFeatures: Int, nLabels: Int): Array[String] = {
    val features = getFeatureNames(nFeatures).mkString(" ")
    val labels = (1 to nLabels).map(i => s"$i:-1 |Y _$i")
    val dummyLabels = Seq(
      "2147483648:0 |y _neg_",
      "2147483649:-1 |y _pos_"
    )

    Array(s"shared |X $features") ++ dummyLabels ++ labels
  }

  private def getTrainedModel(
      nLabels: Int,
      bits: Int,
      trainingData: Array[String],
      initialWeight: Double
  ): ModelSource = {
    val modelFile = tmpFile()
    val params = vwArgs(modelFile, nLabels, bits, initialWeight)
    val learner = VWLearners.create[VWActionScoresLearner](params)
    learner.learn(trainingData)
    learner.close()
    ExternalSource(Vfs.javaFileToAloha(modelFile))
  }

  private def getModel(
    trainedModel: ModelSource,
    nFeatures: Int,
    nLabels: Int,
    nLabelsQueried: Int
  ): Model[Domain, Option[Map[Label, Double]]] = {
    val (featureNames, features) = featureFns(nFeatures)

    MultilabelModel(
      ModelId(1, "model"),
      featureNames,
      features,
      getLabels(nLabels),
      Option(labelExtractor(getLabels(nLabelsQueried))),
      predictorProducer(trainedModel, nLabels),
      Option.empty[Int],
      Auditor
    )
  }
}
