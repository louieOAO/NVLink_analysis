#!/bin/bash
split="split"
# split=""
loop=1
dirname=exp/loop1meshtest

if [ ! -d "$dirname" ];then
    mkdir $dirname
fi

echo "Start monitor"
./monitor 0,1,2,3,4,5,6,7 > $dirname/${split}nvlink.txt &
nvlink_pid=$!

echo "Start ncclReduce"
../ncclReduce $loop $split> $dirname/${split}trace.txt &
trace_pid=$!     
wait ${trace_pid}  
kill -9 ${nvlink_pid}
echo "monitor finish."
if [ $? -eq 0 ]; then
    echo "ncclReduce finish."
else
    echo "ncclReduce Fail!"
fi

python count.py $dirname/${split}nvlink.txt > $dirname/${split}total.txt