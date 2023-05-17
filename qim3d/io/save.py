class DataSaver:
    def __init__(self):
        self.verbose = False

    def save(self, path, data):
        raise NotImplementedError("Save is not implemented yet")


def save(path, data):
    return DataSaver().save(path, data)
