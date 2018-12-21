from time import time
class TicToc:
    def __init__(self):
        self.tic()

    def tic(self):
        self.t = time()

    def toc(self):
        e_time = time()-self.t
        print('Elapsed time {:.2f}'.format(e_time))
        return e_time