import pyrealsense2 as rs
import numpy as np
import cv2
import json

def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    return aligned_depth_frame, color_frame

def save_camera_parameters(color_frame, aligned_depth_frame):
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    camera_parameters = {
        'fx': intr.fx, 'fy': intr.fy,
        'ppx': intr.ppx, 'ppy': intr.ppy,
        'height': intr.height, 'width': intr.width,
        'depth_scale': depth_scale
    }

    with open('camera_parameters.json', 'w') as fp:
        json.dump(camera_parameters, fp)

def save_click_data(x, y, dis, camera_coordinate):
    data = {
        'x': x,
        'y': y,
        'distance': dis,
        'camera_coordinate': camera_coordinate
    }

    with open('click_data.json', 'a') as fp:
        json.dump(data, fp)
        fp.write('\n')  # 每次保存数据后换行

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    img, aligned_depth_frame, intr, pipeline = param

    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)

        dis = aligned_depth_frame.get_distance(x, y)
        print("distance:", dis)

        # Convert pixel coordinates to list of floats
        pixel = [float(x), float(y)]

        # Perform deprojection
        try:
            camera_coordinate = rs.rs2_deproject_pixel_to_point(intr, pixel, dis)
            print("camera_coordinate =", camera_coordinate)
            print("x, y =", x, y)

            # Save click data
            save_click_data(x, y, dis, camera_coordinate)

            # Save RGB image
            cv2.imwrite(f'rgb_at_click_{x}_{y}.jpg', img)

            # Convert aligned depth frame to numpy array
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            # Save depth image
            cv2.imwrite(f'depth_at_click_{x}_{y}.png', depth_image)

        except Exception as e:
            print(f"Error in deprojection: {e}")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        aligned_depth_frame, color_frame = get_aligned_frames(pipeline, align)
        save_camera_parameters(color_frame, aligned_depth_frame)

        rgb = np.asanyarray(color_frame.get_data())
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, (rgb, aligned_depth_frame, profile, pipeline))
        cv2.imshow("image", rgb)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
