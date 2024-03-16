import numpy as np

# Jinping Reader
import  xml.dom.minidom
from io import StringIO
class JinpingReader():
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
# JUNO Reader
class JUNOReader():
    def __init__(self):
        pass
    def read(self, file, filetype):
        if filetype=='abs':
            self.abs = np.genfromtxt(file, dtype=[('E','<f8'), ('Eunit','S3'), ('abs','<f8'), ('absunit','S3')])
        elif filetype=='scale':
            self.abs_scale = 4000/2651.815
        elif filetype=='rindex':
            self.rindex = np.genfromtxt(file, dtype=[('E','<f8'), ('Eunit','S3'), ('rindex','<f8')])
# SK Reader
class SKReader():
    def __init__(self):
        pass
    def read(self, file):
        pass
    def readPMTQE(self, file='SK/QETable.dat'):
        self.pmtQE = np.genfromtxt(file, dtype=[('lambda', '<f8'), ('QE', '<f8')], skip_header=6)
