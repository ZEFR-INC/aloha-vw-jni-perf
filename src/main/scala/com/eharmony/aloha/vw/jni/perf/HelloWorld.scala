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

import scala.collection.immutable
import scala.util.Random

@State(Scope.Thread)
class HelloWorld {
  // TODO: vary: nFeatures, nLabels, nLabelsQueried, bits
  // TODO: JMH docs, warmups, multiple iterations on multiple JVMs
  // TODO: be wary of JVM caching, set this initial_weight 0.000001 randomly

  import HelloWorld._

  @Param(Array("1", "2", "10", "50", "100", "200"))
  var nLabelsQueried: Int = _

  private var model: Model[Domain, Option[Map[Label, Double]]] = _

  @Setup
  def prepare(): Unit = {
    val nFeatures = 20
    val nLabels = 200
    val bits = 22
    model = getModel(nFeatures, nLabels, nLabelsQueried, bits)
  }

  @Benchmark
  def helloWorld(): Unit = {
    model(null)
  }

  @TearDown
  def tearDown(): Unit = {
    model.close()
  }
}

object HelloWorld {
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
  def vwArgs(dest: java.io.File, nLabels: Int, bits: Int): String = {
    val random = new Random()
    val sign = if (random.nextFloat() > 0.5) 1 else -1
    val initialWeight = 1e-6 + sign * random.nextFloat() * 1e-7

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

  private def tmpFile() = {
    val f = File.createTempFile(classOf[HelloWorld].getSimpleName + "_", ".vw.model")
    f.deleteOnExit()
    f
  }

  private def getTrainedModel(nLabels: Int, bits: Int, trainingData: Array[String]): ModelSource = {
    val modelFile = tmpFile()
    val params = vwArgs(modelFile, nLabels, bits)
    println(params)
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

  def getModel(
    nFeatures: Int,
    nLabels: Int,
    nLabelsQueried: Int,
    bits: Int
  ): Model[Domain, Option[Map[Label, Double]]] = {
    val trainingData: Array[String] = vwTrainingExample(nFeatures, nLabels)
    val trainedModel: ModelSource = getTrainedModel(nLabels, bits, trainingData)

    val (featureNames, features) = featureFns(nFeatures)
    val featureFuctions: Vector[GenAggFunc[Domain, Sparse]] = features

    MultilabelModel(
      ModelId(1, "model"),
      featureNames,
      featureFuctions,
      getLabels(nLabels),
      Option(labelExtractor(getLabels(nLabelsQueried))),
      predProd(trainedModel),
      Option.empty[Int],
      Auditor
    )
  }
}
