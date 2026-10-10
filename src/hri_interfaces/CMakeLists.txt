cmake_minimum_required(VERSION 3.8)
project(hri_interfaces)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceDetection.msg"
  "msg/FaceDetectionArray.msg"
  "msg/LipMovement.msg"
  "msg/LipMovementArray.msg"
  "msg/SpeechSegment.msg"
  "msg/TranscriptionResult.msg"
  "msg/VoiceIdentity.msg"
  "msg/FusedIdentity.msg"
  "srv/RAGQuery.srv"
  "srv/EnrollPerson.srv"
  "action/EnrollSession.action"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
