monitor : monitor.c
	gcc monitor.c -o monitor /usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs/libnvidia-ml.so

clean:
	rm monitor 
