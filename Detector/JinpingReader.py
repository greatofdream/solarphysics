import  xml.dom.minidom
import numpy as np
from io import StringIO
class reader():
    def __init__(self):
        pass
    def read(self, file):
        dom = xml.dom.minidom.parse(file)
        root = dom.documentElement
        attrs = root.getElementsByTagName('matrix')
        self.rindex = np.array([float(i) for i in attrs[0].getAttribute('values').replace('*eV','').split()]).reshape((-1,2))
        self.abs = np.array([float(i) for i in attrs[1].getAttribute('values').replace('*eV','').replace('*m','').split()]).reshape((-1,2))
        self.abs[:,1] *= 100
        self.ray = np.array([float(i) for i in attrs[2].getAttribute('values').replace('*eV','').replace('*m','').split()]).reshape((-1,2))
        self.ray[:,1] *= 100
        self.mie = np.array([float(i) for i in attrs[3].getAttribute('values').replace('*eV','').replace('*m','').split()]).reshape((-1,2))
        self.mie[:,1] *= 100