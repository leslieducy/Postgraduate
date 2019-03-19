from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
import re
import os

all_permission_list = set()

sum_repeat = 0
def get_apis(path):
    app = apk.APK(path)
    cs = [cc for cc in app.get_services()]
    return cs

def writeToTxt(filename, content, outfilename):
    # for cont in content:
    #     perm = cont
    #     if perm in all_permission_list:
    #         print("yes")
    #         print(perm)
    #     if perm not in all_permission_list:
    #         all_permission_list.add(perm)
    fm = open(outfilename, 'a', encoding='utf-8')
    fm.write(filename + ":")
    for cont in content:
        perm = cont.split('.')[-1]
        if perm not in all_permission_list:
            all_permission_list.add(perm)
        else:
            global sum_repeat
            sum_repeat += 1
        fm.write(perm)
        fm.write(",")
    fm.write("\n")
    fm.close()

# dirname_well = "G:/pythonProject/appclassify/testapp/"
dirname_well = "G:/pythonProject/appclassify/normalapk/"
dirname_bad = "G:/pythonProject/appclassify/virusesapk/"
out_well = "wellapi.txt"
out_bad = "badapi.txt"
# 输出所有正常的apk权限数据到txt文件中
if os.path.exists(out_well):
    os.remove(out_well)
files = os.listdir(dirname_well)
for filename in files:
    path = dirname_well + filename
    content = get_apis(path)
    writeToTxt(filename,content, out_well)
    print(filename+"服务写入完成！")
print("**************************************正常apk统计完成！")
# 输出所有恶意的apk权限数据到txt文件中
if os.path.exists(out_bad):
    os.remove(out_bad)
files = os.listdir(dirname_bad)
for filename in files:
    path = dirname_bad + filename
    try:
        content = get_apis(path)
        if len(content) == 0 or content[0] == None:
            print(filename+"无效文件！")
            continue
    except:
        print(filename+"无效文件！")
    writeToTxt(filename,content, out_bad)
    print(filename+"服务写入完成！")
print("**************************************恶意apk统计完成！")
# 输出所有统计的权限值
fm = open("all_service.txt", 'w', encoding='utf-8')
for cont in all_permission_list:
    fm.write(cont)
    fm.write("\n")
fm.close()
print("**************************************所有服务统计完成！")
print(sum_repeat)