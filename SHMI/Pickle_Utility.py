import pickle

def write(data, outfile):
        f = open(outfile, "w+b")
        pickle.dump(data, f)
        f.close()
		
def read(filename):
        f = open(filename,'rb')
        data = pickle.load(f,encoding='latin-1')
        f.close()
        return data