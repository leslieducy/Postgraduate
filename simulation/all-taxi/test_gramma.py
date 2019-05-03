import datetime as dati
dat_no = '2013-10-1'
time = '1:09:09'
timeT = dati.datetime.strptime((dat_no + time), '%Y-%m-%d%H:%M:%S')
print(timeT)