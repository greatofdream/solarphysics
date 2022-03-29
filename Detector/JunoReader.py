import numpy as np
class reader():
    def __init__(self):
        pass
    def read(self, file, type):
        if type=='abs':
            self.abs = np.genfromtxt(file, dtype=[('E','<f8'), ('Eunit','S3'), ('abs','<f8'), ('absunit','S3')])
        elif type=='scale':
            self.abs_scale = 4000/2651.815
        elif type=='rindex':
            self.rindex = np.genfromtxt(file, dtype=[('E','<f8'), ('Eunit','S3'), ('rindex','<f8')])