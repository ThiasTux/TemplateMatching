import numpy as np
import progressbar


def extract_binary_data(input_path):
    with open(input_path, 'rb') as input_file:
        with open(input_path.replace(".txt", "_new.txt"), 'w') as output_file:
            header = [None for _ in range(3)]
            byte_counter = 0
            data = None
            bar = progressbar.ProgressBar(max_value=1400000)
            i = 0
            while True:
                header[0] = input_file.read(1)
                if not header[0]:
                    break
                header[1] = input_file.read(1)
                header[2] = input_file.read(1)
                if header[0].decode("utf-8") == "D" and header[1].decode("utf-8") == "X" and header[2].decode(
                        "utf-8") == "X":
                    pck_counter = int.from_bytes(input_file.read(4), byteorder='little', signed=False)
                    raw_values = np.zeros((13))
                    for j in range(9):
                        raw_values[j] = int.from_bytes(input_file.read(2), byteorder='little', signed=True)
                    for j in range(9, 13):
                        raw_values[j] = int.from_bytes(input_file.read(2), byteorder='little', signed=True) / 10000.0
                    checksum = input_file.read(2)
                    output_file.write("{} {} {}\n".format(pck_counter, 0, " ".join([str(v) for v in raw_values])))
                    i += 1
                    bar.update(i)
            bar.finish()


if __name__ == '__main__':
    extract_binary_data("/home/mathias/Documents/Datasets/BeachVolleyball/hand_85CB.txt")
