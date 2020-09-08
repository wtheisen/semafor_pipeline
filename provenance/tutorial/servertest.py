import os
import sys
import numpy as np
def fun1():
    from numpySocketServer import NumpySocket
    import cv2
    imnames = ['/home/jbrogan4/Pictures/0620171632_Pano.jpg','/home/jbrogan4/Pictures/0620171630_HDR.jpg']*2

    npSocket2 = None
    for imname in imnames:
        npSocket = NumpySocket()
        npSocket.startServer('localhost', 9999)
        npSocket2 = None
        im = cv2.imread(imname)
        npSocket.sendNumpy(im)
        #npSocket.socket.shutdown(1)
        npSocket.socket.close()
        if npSocket2 is None:
            npSocket2 = NumpySocket()
            print('starting client')
            npSocket2.startClient(8312)
            print('started')
        retim = npSocket2.recieveNumpy()
        print('got image', retim.shape)
        #npSocket2.socket.shutdown(1)
        npSocket2.socket.close()
        #import matplotlib.pyplot as plt
        #plt.imshow(retim)
        #plt.show()

fun1()