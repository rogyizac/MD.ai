import hashlib
import os


def generateModelVersion(filename):
    md5_hash = hashlib.md5()
    with open(filename, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)

        return md5_hash.hexdigest()


def scanDirectory(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = scanDirectory(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def getLatestModel(model='', baseDir='model/'):
    subfolders, files = scanDirectory(baseDir + model, [".pkl"])
    return max(files, key=os.path.getctime)
