name := """aloha-vw-jni-perf"""
scalaVersion := "2.11.8"

version := "1.0-SNAPSHOT"

libraryDependencies ++= Seq(
  "com.eharmony" %% "aloha-core" % "5.0.1-SNAPSHOT",
  "com.eharmony" %% "aloha-vw-jni" % "5.0.1-SNAPSHOT"
)

enablePlugins(JmhPlugin)
