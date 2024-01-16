import numpy , math


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    def populate(samples, h):
        k = []
        sum = 0;
        for i in samples:
            k.append([i, findValue(i)])
            sum+= i*findValue(i)
        print('sum is ', sum)
        return k

    def findValue(x):
        val = 0
        for i in samples:
            val += (((2*math.pi*h**2)**(-1))*math.e**(-1*(((x-i)**2)/(2*(h**2)))))/100
        return val

    res = numpy.array(populate(numpy.sort(samples), h))

    return res
    
