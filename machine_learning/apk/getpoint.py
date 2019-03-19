# from androguard.core.bytecodes import apk, dvm
from androguard.core.bytecodes import apk
# from androguard.core.analysis import analysis
import re
import os

all_permission_list = set()

def get_permissions(path):
    try:
        app = apk.APK(path)
        permission = app.get_permissions()
    except:
        permission = []
    return permission

def writeToTxt(filename, content, outfilename):
    fm = open(outfilename, 'a')
    fm.write(filename + ":")
    for cont in content:
        perm = cont.split('permission')[-1]
        if perm not in all_permission_list:
            all_permission_list.add(perm)
        fm.write(perm)
        fm.write(",")
    fm.write("\n")
    fm.close()

dirname_well = "G:/pythonProject/appclassify/normalapk/"
dirname_bad = "G:/pythonProject/appclassify/virusesapk/"
out_well = "well.txt"
out_bad = "bad.txt"
# 输出所有正常的apk权限数据到txt文件中
if os.path.exists(out_well):
    os.remove(out_well)
files = os.listdir(dirname_well)
for filename in files:
    path = dirname_well + filename
    content = get_permissions(path)
    writeToTxt(filename,content, out_well)
    print(filename+"权限写入完成！")
print("**************************************正常apk统计完成！")
# 输出所有恶意的apk权限数据到txt文件中
if os.path.exists(out_bad):
    os.remove(out_bad)
files = os.listdir(dirname_bad)
for filename in files:
    path = dirname_bad + filename
    content = get_permissions(path)
    if len(content) == 0 or content[0] == None:
        print(filename+"无效文件！")
        continue
    writeToTxt(filename,content, out_bad)
    print(filename+"权限写入完成！")
print("**************************************恶意apk统计完成！")
# 输出所有统计的权限值
fm = open("all_permission.txt", 'w')
for cont in all_permission_list:
    fm.write(cont)
    fm.write("\n")
fm.close()
print("**************************************所有权限完成！")