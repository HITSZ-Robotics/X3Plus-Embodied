import sys
import pyrealsense2 as rs


pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
#cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
cfg = pipeline.start(config)




device = cfg.get_device()
name = device.get_info(rs.camera_info.name)
print(name)
profile = cfg.get_stream(rs.stream.depth)
profile1 = cfg.get_stream(rs.stream.color)
intr = profile.as_video_stream_profile().get_intrinsics()
intr1 = profile1.as_video_stream_profile().get_intrinsics()
extrinsics = profile1.get_extrinsics_to(profile)
print(extrinsics)
print("深度传感器内参：", intr)
print("RGB相机内参:", intr1)
pipeline.stop()
