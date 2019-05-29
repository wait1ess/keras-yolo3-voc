#-*-coding:utf-8-*-
import os
 
path = "./**"
 
 
i=1
for item in os.listdir(path):
    old_name = os.path.join(path,item)
    new_name = os.path.join(path,(str(i).zfill(6)+'.jpg'))
    os.rename(old_name, new_name)
    i+=1
