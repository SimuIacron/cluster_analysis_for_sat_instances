import csv
import os


# exports a csv containing the data, with the header appended as the first row
def write_to_csv(file, header, data):

    data = [header] + data

    with open(os.environ['TEXPATH'] + file + '.csv', 'w', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='&')
        my_writer.writerows(data)
