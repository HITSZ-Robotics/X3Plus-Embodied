#!/user/bin/env python
# coding: utf-8

import socket
import os
import json

def download(file_name):  #filename has ''   !!!!!!!

    ip_port = ('192.168.43.220',8888)
 
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.connect(ip_port)
  #  base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/" #客户端文件位置
   # base_path=r"/home/yahboom/yahboomcar_ws/src/robot/data/"
    base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/"


    op="Download"
    sk.send(op.encode())
    print(sk.recv(1024).decode())
    sk.send(file_name.encode())    
    
    file_inf = sk.recv(1024).decode()
    # 获取请求方法、文件名、文件大小
    file_name, file_size = file_inf.split('|')  # 分割 文件名和大小    
    
    # 已经接收文件的大小
    recv_size = 0
    # 上传文件路径拼接
    file_dir = os.path.join(base_path, file_name)
    f = open(file_dir, 'wb')
    Flag = True
    while Flag:
        # 未上传完毕，
        if int(file_size) > recv_size:
            # 最多接收1024，可能接收的小于1024
            data = sk.recv(1024)
            recv_size += len(data)
            # 写入文件
            f.write(data)
        # 上传完毕，则退出循环
        else:
            recv_size = 0
            Flag = False
    msg = "Download successed."
    sk.send(msg.encode())
    f.close()
    sk.close()
    

def upload(file_name):
  
    ip_port = ('192.168.43.220',8888)
    
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.connect(ip_port)
   # base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/" #客户端文件位置
    base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/" 

    path = base_path + file_name
    op="Upload"
    sk.send(op.encode())
    print(sk.recv(1024).decode())
        # 客户端输入要上传文件的路径
        # 获取文件大小
    file_size = os.path.getsize(path)
    # 发送文件名 和 文件大小u
    Informf = (str(file_name) +'|'+ str(file_size))
    sk.send(Informf.encode())
    # 为了防止粘包，将文件名和大小发送过去之后，等待服务端收到，直到从服务端接受一个信号（说明服务端已经收到）
    print(sk.recv(1024).decode())
    #指定了缓冲区的大小为1024字节，因此需要对于文件循环读取，直到文件的数据填满了一个缓冲区，此时将缓冲区数据发送出去，
    # 继续读取下一部分文件；或是当缓冲区未填满，而文件读取完毕，此时应当将这个未满的缓冲区发送给服务器。
    send_size = 0
    f = open(path, 'rb')
    Flag = True
    while Flag:
        if send_size + 1024 > file_size:
            data = f.read(file_size - send_size)
            Flag = False
        else:
            data = f.read(1024)
            send_size += 1024
        sk.send(data)
    msg = sk.recv(1024)
    print(msg.decode())
    f.close()
    print()
    sk.close()
    print(str(file_name)+' download sucess')


#def UD_finish():
#    serverName = 'localhost_up'
#    serverPort = 12000
#    ip_port = ('192.168.253.1',8888)
#    ip_port = (serverName, serverPort)
#    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    sk.connect(ip_port)
#    
#
#    op="Finish"
#    sk.send(op.encode())
#    sk.close()
    
