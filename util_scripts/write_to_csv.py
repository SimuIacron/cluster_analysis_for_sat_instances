import csv
import os


def write_to_csv(file, header, data):

    data = [header] + data

    with open(os.environ['TEXPATH'] + file + '.csv', 'w', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='&')
        my_writer.writerows(data)