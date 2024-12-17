import json
base_path = "/home/zzl/Desktop/master/data/"
def write_point(x,y,file_name):
    data={"x":x,"y":y}
    with open (base_path+file_name,"w") as f:
          json.dump(data,f)
     
def write_pickpoint(x,y):
     write_point(x,y,"pickpoint.json")

def write_navigate(x, y, a ,filename):
    data={"x":x,"y":y,"a":a}
    with open (base_path+filename,"w") as f:
          json.dump(data,f)

def write_navigatepoint(x, y, a):
     write_navigate(x, y, a,"navigatepoint.json")