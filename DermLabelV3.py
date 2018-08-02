# Import modules
import os
import tkinter as tk
from google_images_download import google_images_download
from PIL import ImageTk, Image


# Function definitions
def redness(rating):
    global ratings

    # Assign user selected rating
    ratings[0] = rating

    # Make number glow blue to indicate selection
    for x in range(10):
        red_button[x]['fg'] = 'black'
        red_button[x]['font'] = 'Arial 18'
    red_button[rating]['fg'] = 'blue'
    red_button[rating]['font'] = 'Arial 18 bold'


def texture(rating):
    global ratings

    # Assign user selected rating
    ratings[1] = rating

    # Make number glow blue to indicate selection
    for x in range(10):
        texture_button[x]['fg'] = 'black'
        texture_button[x]['font'] = 'Arial 18'
    texture_button[rating]['fg'] = 'blue'
    texture_button[rating]['font'] = 'Arial 18 bold'


def evenness(rating):
    global ratings

    # Assign user selected rating
    ratings[2] = rating

    # Make number glow blue to indicate selection
    for x in range(10):
        even_button[x]['fg'] = 'black'
        even_button[x]['font'] = 'Arial 18'
    even_button[rating]['fg'] = 'blue'
    even_button[rating]['font'] = 'Arial 18 bold'


def refresh():
    global ratings
    global filename

    # Reset rating buttons
    for x in range(10):
        red_button[x]['fg'] = 'black'
        red_button[x]['font'] = 'Arial 18'
        texture_button[x]['fg'] = 'black'
        texture_button[x]['font'] = 'Arial 18'
        even_button[x]['fg'] = 'black'
        even_button[x]['font'] = 'Arial 18'
        ratings = [None, None, None]

    # Load next image
    if file_paths:
        file_title['text'] = file_paths[-1][-40:] + ' (%d remaining)' % len(file_paths)
        filename = file_paths.pop()
        img_new = Image.open('images/' + filename)
        img_new.thumbnail((w//2, h//2))
        img_new = ImageTk.PhotoImage(img_new)
        image['image'] = img_new
        image.image = img_new
        image.grid(row=2, column=0, columnspan=10)

    # Remove buttons when no more images left
    else:
        skip.destroy()
        more.destroy()
        submit.destroy()
        file_title['text'] = filename[-40:] + ' (0 remaining)'
        finished = tk.Label(root, text='Done!', fg='black', font='Helvetica 18 bold')
        finished.grid(row=10, column=0, columnspan=10)


def skip():
    global ratings
    global filename

    # Write "no rating" to file for skipped images
    image_url = url()
    with open('ratings.txt', 'a') as file:
        file.write('_, _, _, ' + str(filename) + ', ' + image_url + '\n')
    print(str(filename) + ': no rating')

    # Reset buttons and load next image
    refresh()


def url():
    # Identify log file and image number
    log_file = 'logs/' + filename.split('/')[0] + '.txt'
    image_num = int(filename.split('/')[1].split(' ')[0][:-1]) - 1

    # Get image url for Google image scraping
    with open(log_file, 'r') as file:
        image_url = list(file)[10 * image_num + 6]
    return image_url.split(' ')[-1][1:-3]


def submit():
    global ratings
    global filename

    # Only write ratings to file when all ratings have been chosen
    if None not in ratings:
        image_url = url()
        with open('ratings.txt', 'a') as file:
            file.write('%d, %d, %d, ' % tuple(ratings) + str(filename) + ', ' + image_url + '\n')
        print(str(filename) + ': %d, %d, %d' % tuple(ratings))

        # Reset buttons and load next image
        refresh()


def more():
    global file_paths

    # Get image url for Google image scraping
    image_url = url()

    # Set image download folder
    downloads = os.path.dirname(os.path.abspath(__file__)) + '/images'

    # Scrape Google images
    response = google_images_download.googleimagesdownload()
    arguments = {'similar_images': image_url, 'limit': 20, 'format': 'jpg',
                 'output_directory': downloads, 'color_type': 'full-color',
                 'type': 'face', 'extract_metadata': True}
    path = response.download(arguments)

    # List of new image filenames
    new_images = [list(path.keys())[0] + '/' + x for x in
                  os.listdir('images/' + list(path.keys())[0]) if x[-3:] == 'jpg']
    new_images = sorted(new_images, reverse=True)

    # Append new image filenames to current list of paths
    file_paths = new_images + file_paths

    # Update remaining number of images to label
    file_title['text'] = file_paths[-1][-40:] + ' (%d remaining)' % len(file_paths)


# Get filenames
file_paths = [y+'/'+x for y in os.listdir('images') if y[0] != '.'
              for x in os.listdir('images/'+y) if x[-4:] == '.jpg']
file_paths = sorted(file_paths, reverse=True)
if os.path.isfile('ratings.txt'):
    with open('ratings.txt', 'r') as image_name:
        file_paths = file_paths[:file_paths.index(list(image_name)[-1].split(',')[3][1:])]

# Preallocate ratings
ratings = [None, None, None]

# Define root widget (parent of all subwidgets)
root = tk.Tk()

# Get screen pixel dimensions
w, h = root.winfo_screenwidth(), root.winfo_screenheight()

# Title
title = tk.Label(root, text='Dermatology Image Labeler', fg='black', font='Helvetica 24 bold')
title.grid(row=0, column=0, columnspan=10)

# Filename
file_title = tk.Label(root, text=file_paths[-1][-40:] + ' (%d remaining)' % len(file_paths),
                      fg='black', font='Helvetica 18')
file_title.grid(row=1, column=0, columnspan=10)

# Image
filename = file_paths.pop()
img = Image.open('images/' + filename)
img.thumbnail((w//2, h//2))
img = ImageTk.PhotoImage(img)
image = tk.Label(root, height=h//2, width=h//2, image=img)
image.grid(row=2, column=0, columnspan=10)

# Redness
red_title = tk.Label(root, text='Redness', fg='black', font='Helvetica 18')
red_title.grid(row=3, column=0, columnspan=10)
red_button = []
for i in range(10):
    red_button += [tk.Button(root, text=str(i), width=h//250, fg='black', font='Arial 18',
                             command=lambda rating=i: redness(rating))]
    red_button[i].grid(row=4, column=i)

# Texture
texture_title = tk.Label(root, text='Texture', fg='black', font='Helvetica 18')
texture_title.grid(row=5, column=0, columnspan=10)
texture_button = []
for i in range(10):
    texture_button += [tk.Button(root, text=str(i), width=h//250, fg='black', font='Arial 18',
                                 command=lambda rating=i: texture(rating))]
    texture_button[i].grid(row=6, column=i)

# Evenness
even_title = tk.Label(root, text='Evenness', fg='black', font='Helvetica 18')
even_title.grid(row=7, column=0, columnspan=10)
even_button = []
for i in range(10):
    even_button += [tk.Button(root, text=str(i), width=h//250, fg='black', font='Arial 18',
                              command=lambda rating=i: evenness(rating))]
    even_button[i].grid(row=8, column=i)

# Gap
gap = tk.Label(root, bg='white')
gap.grid(row=9, column=0, columnspan=10)

# Skip button
skip = tk.Button(root, text='Skip', font='Helvetica 18 bold', command=skip)
skip.grid(row=10, column=1, columnspan=2)

# More like this button
more = tk.Button(root, text='More Like This', font='Helvetica 18 bold', command=more)
more.grid(row=10, column=3, columnspan=4)

# Submit button
submit = tk.Button(root, text='Submit', font='Helvetica 18 bold', command=submit)
submit.grid(row=10, column=7, columnspan=2)

# Make GUI interface appear
root.mainloop()
