import os


def find_files(folder, file_type):
    files_ = []
    list = [i for i in os.listdir(folder)]
    for i in range(0, len(list)):
        path = os.path.join(folder, list[i])
        if os.path.isdir(path):
            files_.extend(find_files(path, file_type))
        if not os.path.isdir(path):
            if path.lower().endswith(file_type):
                files_.append(path)
    return files_
