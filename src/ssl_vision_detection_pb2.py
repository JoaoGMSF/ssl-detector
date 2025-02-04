# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ssl_vision_detection.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ssl_vision_detection.proto',
  package='',
  syntax='proto2',
  serialized_pb=_b('\n\x1assl_vision_detection.proto\"x\n\x11SSL_DetectionBall\x12\x12\n\nconfidence\x18\x01 \x02(\x02\x12\x0c\n\x04\x61rea\x18\x02 \x01(\r\x12\t\n\x01x\x18\x03 \x02(\x02\x12\t\n\x01y\x18\x04 \x02(\x02\x12\t\n\x01z\x18\x05 \x01(\x02\x12\x0f\n\x07pixel_x\x18\x06 \x02(\x02\x12\x0f\n\x07pixel_y\x18\x07 \x02(\x02\"\x97\x01\n\x12SSL_DetectionRobot\x12\x12\n\nconfidence\x18\x01 \x02(\x02\x12\x10\n\x08robot_id\x18\x02 \x01(\r\x12\t\n\x01x\x18\x03 \x02(\x02\x12\t\n\x01y\x18\x04 \x02(\x02\x12\x13\n\x0borientation\x18\x05 \x01(\x02\x12\x0f\n\x07pixel_x\x18\x06 \x02(\x02\x12\x0f\n\x07pixel_y\x18\x07 \x02(\x02\x12\x0e\n\x06height\x18\x08 \x01(\x02\"\xd9\x01\n\x12SSL_DetectionFrame\x12\x14\n\x0c\x66rame_number\x18\x01 \x02(\r\x12\x11\n\tt_capture\x18\x02 \x02(\x01\x12\x0e\n\x06t_sent\x18\x03 \x02(\x01\x12\x11\n\tcamera_id\x18\x04 \x02(\r\x12!\n\x05\x62\x61lls\x18\x05 \x03(\x0b\x32\x12.SSL_DetectionBall\x12*\n\rrobots_yellow\x18\x06 \x03(\x0b\x32\x13.SSL_DetectionRobot\x12(\n\x0brobots_blue\x18\x07 \x03(\x0b\x32\x13.SSL_DetectionRobotB8Z6github.com/RoboCup-SSL/ssl-simulation-protocol/pkg/sim')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_SSL_DETECTIONBALL = _descriptor.Descriptor(
  name='SSL_DetectionBall',
  full_name='SSL_DetectionBall',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='confidence', full_name='SSL_DetectionBall.confidence', index=0,
      number=1, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='area', full_name='SSL_DetectionBall.area', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='x', full_name='SSL_DetectionBall.x', index=2,
      number=3, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y', full_name='SSL_DetectionBall.y', index=3,
      number=4, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='z', full_name='SSL_DetectionBall.z', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pixel_x', full_name='SSL_DetectionBall.pixel_x', index=5,
      number=6, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pixel_y', full_name='SSL_DetectionBall.pixel_y', index=6,
      number=7, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=30,
  serialized_end=150,
)


_SSL_DETECTIONROBOT = _descriptor.Descriptor(
  name='SSL_DetectionRobot',
  full_name='SSL_DetectionRobot',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='confidence', full_name='SSL_DetectionRobot.confidence', index=0,
      number=1, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='robot_id', full_name='SSL_DetectionRobot.robot_id', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='x', full_name='SSL_DetectionRobot.x', index=2,
      number=3, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='y', full_name='SSL_DetectionRobot.y', index=3,
      number=4, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='orientation', full_name='SSL_DetectionRobot.orientation', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pixel_x', full_name='SSL_DetectionRobot.pixel_x', index=5,
      number=6, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pixel_y', full_name='SSL_DetectionRobot.pixel_y', index=6,
      number=7, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='SSL_DetectionRobot.height', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=153,
  serialized_end=304,
)


_SSL_DETECTIONFRAME = _descriptor.Descriptor(
  name='SSL_DetectionFrame',
  full_name='SSL_DetectionFrame',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame_number', full_name='SSL_DetectionFrame.frame_number', index=0,
      number=1, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='t_capture', full_name='SSL_DetectionFrame.t_capture', index=1,
      number=2, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='t_sent', full_name='SSL_DetectionFrame.t_sent', index=2,
      number=3, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='camera_id', full_name='SSL_DetectionFrame.camera_id', index=3,
      number=4, type=13, cpp_type=3, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='balls', full_name='SSL_DetectionFrame.balls', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='robots_yellow', full_name='SSL_DetectionFrame.robots_yellow', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='robots_blue', full_name='SSL_DetectionFrame.robots_blue', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=307,
  serialized_end=524,
)

_SSL_DETECTIONFRAME.fields_by_name['balls'].message_type = _SSL_DETECTIONBALL
_SSL_DETECTIONFRAME.fields_by_name['robots_yellow'].message_type = _SSL_DETECTIONROBOT
_SSL_DETECTIONFRAME.fields_by_name['robots_blue'].message_type = _SSL_DETECTIONROBOT
DESCRIPTOR.message_types_by_name['SSL_DetectionBall'] = _SSL_DETECTIONBALL
DESCRIPTOR.message_types_by_name['SSL_DetectionRobot'] = _SSL_DETECTIONROBOT
DESCRIPTOR.message_types_by_name['SSL_DetectionFrame'] = _SSL_DETECTIONFRAME

SSL_DetectionBall = _reflection.GeneratedProtocolMessageType('SSL_DetectionBall', (_message.Message,), dict(
  DESCRIPTOR = _SSL_DETECTIONBALL,
  __module__ = 'ssl_vision_detection_pb2'
  # @@protoc_insertion_point(class_scope:SSL_DetectionBall)
  ))
_sym_db.RegisterMessage(SSL_DetectionBall)

SSL_DetectionRobot = _reflection.GeneratedProtocolMessageType('SSL_DetectionRobot', (_message.Message,), dict(
  DESCRIPTOR = _SSL_DETECTIONROBOT,
  __module__ = 'ssl_vision_detection_pb2'
  # @@protoc_insertion_point(class_scope:SSL_DetectionRobot)
  ))
_sym_db.RegisterMessage(SSL_DetectionRobot)

SSL_DetectionFrame = _reflection.GeneratedProtocolMessageType('SSL_DetectionFrame', (_message.Message,), dict(
  DESCRIPTOR = _SSL_DETECTIONFRAME,
  __module__ = 'ssl_vision_detection_pb2'
  # @@protoc_insertion_point(class_scope:SSL_DetectionFrame)
  ))
_sym_db.RegisterMessage(SSL_DetectionFrame)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('Z6github.com/RoboCup-SSL/ssl-simulation-protocol/pkg/sim'))
# @@protoc_insertion_point(module_scope)
