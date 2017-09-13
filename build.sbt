name := """aloha-vw-jni-perf"""

version := "1.0-SNAPSHOT"

libraryDependencies ++= Seq(
  // Add your own project dependencies in the form:
  // "group" % "artifact" % "version"
  // add aloha core, aloha-vw-jni
)

enablePlugins(JmhPlugin)
