import socket
import re
import json
base_path=r"/home/jetson/ROS/X3plus/yahboomcar_ws/src/robot/data/"
def get_local_ip():
    """自动查询本地 IP 地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 使用 Google DNS 服务器来获取本地 IP 地址
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception as e:
        print(f"无法自动获取robot IP 地址: {e}")
        ip = None
    finally:
        s.close()
    return ip

def is_valid_ip(ip):
    """验证 IP 地址格式"""
    pattern = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
    return pattern.match(ip) is not None

def ip_json():
    """手动输入 IP 地址"""
    while True:
        ip = input("请输入 master IP 地址: ")
        if is_valid_ip(ip):
            ip_data={"master_ip": ip }
            with open (base_path+"ip.json","w") as f:
               json.dump(ip_data,f)

            break
        else:
            print("无效的 IP 地址，请重新输入。")

def get_manual_ip():
    """手动输入 IP 地址"""
    while True:
        with open (base_path +"ip.json","r") as f:
            ip_data=json.load(f)
        ip = ip_data["master_ip"]
        if is_valid_ip(ip):
            return ip
        else:
            print("无效的 IP 地址，请重新输入。")
if __name__ == "__main__":
    # 自动获取本地 IP 地址
    local_ip = get_local_ip()
    if local_ip:
        print(f"本地 IP 地址: {local_ip}")
    else:
        print("无法自动获取本地 IP 地址。")

    # 手动输入 IP 地址
    manual_ip = get_manual_ip()
    print(f"手动输入的 IP 地址: {manual_ip}")
