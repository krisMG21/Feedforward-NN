import csv
import os
from PIL import Image

def write_img_to_csv(path, file):
    ''' Searches for 'path' image, reads pixels values and writes them onto user_entries.csv. Also takes
    the first character of the filename as the number is represented inside'''
    foto = Image.open(path)

    datos = list(file[0]) + list(list(zip(*foto.getdata()))[0])
    # print(datos)

    with open('user_entries.csv', 'a', newline='') as archivo:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(archivo)
        writer.writerow(datos) # Use writerow for single list

    foto.close()

def write_all_img_to_csv(path):
    '''In a given directory, searches for '''
    files = os.listdir(path)

    for elem in files:
        if elem.endswith(".png"):
            write_img_to_csv(path+"/"+elem, elem)
            print(f"Delete or move file: {elem}")


def read_img_values(path):
    foto = Image.open(path)

    datos = list(zip(*foto.getdata()))[0]
    # print(datos)

    foto.close()
    return datos

if __name__ == "__main__":
    # print(read_img_values("drawn_num.png"))
    write_all_img_to_csv("Images_for_csv")
