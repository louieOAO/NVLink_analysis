# f1 = open('record.txt', 'r')
# # f0 = open('record_1.txt', 'r')
# # result = open('sort.txt', 'w+')
# # data = f01.readlines() + f23.readlines()
# # data = sorted(data, key = lambda x: int(x.split(' ')[0]))
# # result.writelines(data)

# rxcnt = 0
# txcnt = 0
# gpuid = '1'
# for line in f1.readlines():
#     if line.split(' ')[2] == gpuid and line.split(' ')[5] == 'Tx:':
#         txcnt = txcnt + float(line.split(' ')[6])
#     if line.split(' ')[2] == gpuid and line.split(' ')[5] == 'Rx:':
#         rxcnt = rxcnt + float(line.split(' ')[6])
        




def NVLinkinfo(filename, gpuid):
    f = open(filename, 'r')

    rxcnt = 0
    txcnt = 0
    for line in f.readlines():
        if line.split(' ')[2] == gpuid and line.split(' ')[5] == 'Tx:':
            txcnt = txcnt + float(line.split(' ')[6])
        if line.split(' ')[2] == gpuid and line.split(' ')[5] == 'Rx:':
            rxcnt = rxcnt + float(line.split(' ')[6])
    
    print("inside",filename)
    print("**************************")
    print("GPU id:", str(gpuid))
    print(f'Tx {txcnt/1024/1024} GiB')
    print(f'Rx {rxcnt/1024/1024} GiB')
    print("**************************")


NVLinkinfo('record.txt', '1')
    