import sys
def NVLinkinfo(filename, gpuid):
    f = open(filename, 'r')
    cnt = 0
    rxcnt = [0 for i in range(len(gpuid))]
    txcnt = [0 for i in range(len(gpuid))]
    for line in f.readlines():
        cnt+=1
        try:
            index = int(line.split(' ')[2])
            if index in gpuid and line.split(' ')[5] == 'Tx:':
                txcnt[index] = txcnt[index] + float(line.split(' ')[6])
            if index in gpuid and line.split(' ')[5] == 'Rx:':
                rxcnt[index] = rxcnt[index] + float(line.split(' ')[6])
        except:
            print(cnt)
    
    
    print("inside",filename)
    print("**************************")
    for i in range(len(gpuid)):
        print("GPU id:", i)
        print(f'Tx {txcnt[i]/1024/1024} GiB')
        print(f'Rx {rxcnt[i]/1024/1024} GiB')
    print("**************************")



if len(sys.argv) < 2:
    print('Lack File Name')
    sys.exit()
filename = sys.argv[1]
gpuid = [i for i in range(8)]
NVLinkinfo(filename, gpuid)
    
    