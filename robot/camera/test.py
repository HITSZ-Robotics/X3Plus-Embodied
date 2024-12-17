import camera_data

def request_capture():
    camera_data.set_capture_signal(True)
    print("Capture signal set to True")

if __name__ == "__main__":
    request_capture()
