import hashlib


def hash_name(name):
    return hashlib.md5(name.encode()).hexdigest()
