import socket
import os
import json
from ip import get_manual_ip

master_ip = get_manual_ip()
master_ip = ('{}'.format(master_ip), 8888)

base_path = r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/"

def download(file_name):
    ip_port = master_ip
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sk.connect(ip_port)
    except Exception as e:
        print(f"Connection error: {e}")
        return

    try:
        op = "Download"
        sk.send(op.encode())
        print(sk.recv(1024).decode())
        sk.send(file_name.encode())

        file_inf = sk.recv(1024).decode()
        if '|' not in file_inf:
            raise ValueError("Invalid file information received")

        file_name, file_size = file_inf.split('|')
        file_size = int(file_size)

        recv_size = 0
        file_dir = os.path.join(base_path, file_name)
        
        with open(file_dir, 'wb') as f:
            while recv_size < file_size:
                data = sk.recv(1024)
                if not data:
                    break
                recv_size += len(data)
                f.write(data)
                
        if recv_size == file_size:
            print("Download succeeded.")
            sk.send("Download succeeded.".encode())
        else:
            print("Download failed: incomplete file.")
            sk.send("Download failed.".encode())

    except Exception as e:
        print(f"Failed to download file: {e}")
    finally:
        sk.close()

def upload(file_name):
    ip_port = master_ip
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sk.connect(ip_port)
    except Exception as e:
        print(f"Connection error: {e}")
        return

    path = os.path.join(base_path, file_name)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sk.close()
        return

    try:
        op = "Upload"
        sk.send(op.encode())
        print(sk.recv(1024).decode())

        file_size = os.path.getsize(path)
        informf = f"{file_name}|{file_size}"
        sk.send(informf.encode())
        print(sk.recv(1024).decode())

        send_size = 0
        with open(path, 'rb') as f:
            while send_size < file_size:
                data = f.read(1024)
                if not data:
                    break
                sk.send(data)
                send_size += len(data)

        if send_size == file_size:
            print("Upload succeeded.")
        else:
            print("Upload failed: incomplete file.")
        
        msg = sk.recv(1024)
        print(msg.decode())

    except Exception as e:
        print(f"Failed to upload file: {e}")
    finally:
        sk.close()

# Example usage:
# download('example_file.txt')
# upload('example_file.txt')

#download('zzl.json')    
