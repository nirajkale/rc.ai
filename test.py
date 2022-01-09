import time

for i in range(100):
    time.sleep(0.1)
    print('%d - %d\r'%(i, i+1), end="")