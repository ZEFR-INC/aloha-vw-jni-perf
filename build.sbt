name := """aloha-vw-jni-perf"""

version := "1.0-SNAPSHOT"

libraryDependencies ++= Seq(
  "com.eharmony" % "aloha-core_2.11" % "5.0.1-SNAPSHOT",
  "com.eharmony" % "aloha-vw-jni_2.11" % "5.0.1-SNAPSHOT"
)

enablePlugins(JmhPlugin)
