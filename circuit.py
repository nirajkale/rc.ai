rb = 20000# ohms
rc = 273
vcc = 5 #v
vb = 3.6
ib = vb/rb

gain = 110
ic = ib * gain
v_load = ic * rc

print('IB: ',ib*1000, 'mA')
print('IC: ',ic*1000, 'mA')
print('V Load: ', v_load, 'v')