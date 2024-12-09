def initialize_files(train_f, test_f):
    file1 = open(train_f, "w")
    header1 = (
        'dataclass\tnum_samples\tscenario\tmodel\titeration\taccuracy\tprecision\trecall\tf1_score\ttime_taken\n')
    file1.write(header1)

    file2 = open(test_f, "w")
    header2 = (
        'dataclass\tnum_samples\tscenario\tmodel\titeration\taccuracy\tprecision\trecall\tf1_score\ttime_taken\n')
    file2.write(header2)

    return file1, file2


def append_files(train_f, test_f):
    file1 = open(train_f, "a")
    file2 = open(test_f, "a")

    return file1, file2


def close_files(file1, file2):
    file1.close()
    file2.close()
