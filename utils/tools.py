import os


def find_files(folder):
    files_ = []
    list = [i for i in os.listdir(folder)]
    for i in range(0, len(list)):
        path = os.path.join(folder, list[i])
        if os.path.isdir(path):
            files_.extend(find_files(path))
        if not os.path.isdir(path):
            if path.lower().endswith("pdf"):
                files_.append(path)
    return files_
