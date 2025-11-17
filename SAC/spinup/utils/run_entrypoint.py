import zlib
import pickle
import base64

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    input("Print 1")
    parser.add_argument('encoded_thunk')
    input("Print 1")
    args = parser.parse_args()
    input("Print 1")
    print(args.encoded_thunk)
    input("Print 1")
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    input("Print 1")
    # thunk()