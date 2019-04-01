import argparse
import hashlib

HASH_LIBS = ['md5', 'sha1', 'sha256', 'sha512']
BUFFER_SIZE = 1024 ** 3

parser = argparse.ArgumentParser()
parser.add_argument("FILE", help="File to hash")
parser.add_argument("-a", "--algorithm", help="Hash algorithm to use", choices=HASH_LIBS, default="sha512")
args = parser.parse_args()

alg = getattr(hashlib, args.algorithm)()

with open(args.FILE, 'rb') as input_file:
    buffer_data = input_file.read(BUFFER_SIZE)
    while buffer_data:
        alg.update(buffer_data)
        buffer_data = input_file.read(BUFFER_SIZE)

print(alg.hexdigest())

