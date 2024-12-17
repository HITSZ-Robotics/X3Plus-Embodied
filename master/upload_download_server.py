import socketserver
import os
import json
from ip_get import get_local_ip

master_ip = get_local_ip()
print(f"Master IP address: {master_ip}")
ip_port_ud = (master_ip, 8888)
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path=parent_directory+"/data/"
class MyServerLoad(socketserver.BaseRequestHandler):
    def handle(self):
        print("Server for upload and download has started")
      
        conn = self.request

        while True:
            try:
                op = conn.recv(1024).decode()
                if op == "Upload":
                    self.handle_upload(conn, base_path)
                elif op == "Download":
                    self.handle_download(conn, base_path)
            except Exception as e:
                print(f"An error occurred: {e}")
                break

    def handle_upload(self, conn, base_path):
        conn.send("Upload-ok".encode())
        file_inf = conn.recv(1024).decode()
        file_name, file_size = file_inf.split('|')
        file_size = int(file_size)
        file_dir = os.path.join(base_path, file_name)
        conn.send("get the name".encode())
        try:
            with open(file_dir, 'wb') as f:
                recv_size = 0
                while recv_size < file_size:
                    data = conn.recv(1024)
                    if not data:
                        break
                    recv_size += len(data)
                    f.write(data)
            
            if recv_size == file_size:
                msg = "Upload succeeded."
                print(msg)
                conn.sendall(msg.encode())
            else:
                msg = "Upload failed: incomplete file."
                print(msg)
                conn.sendall(msg.encode())

        except Exception as e:
            print(f"Failed to upload file: {e}")
            conn.sendall("Upload failed.".encode())

    def handle_download(self, conn, base_path):
        conn.send("Download-ok".encode())
        file_name = conn.recv(1024).decode()
        file_path = os.path.join(base_path, file_name)
        
        if not os.path.exists(file_path):
            conn.sendall("File not found.".encode())
            return
        
        file_size = os.path.getsize(file_path)
        informf = f"{file_name}|{file_size}"
        conn.sendall(informf.encode())

        try:
            with open(file_path, 'rb') as f:
                send_size = 0
                while send_size < file_size:
                    data = f.read(1024)
                    if not data:
                        break
                    conn.send(data)
                    send_size += len(data)
            
            if send_size == file_size:
                print(f"Download of {file_name} succeeded.")
            else:
                print(f"Download failed: incomplete file.")
                
        except Exception as e:
            print(f"Failed to download file: {e}")

if __name__ == '__main__':
    instance = socketserver.ThreadingTCPServer(ip_port_ud, MyServerLoad)
    instance.serve_forever()
