# Import modules
import os
import tkinter as tk
from PIL import ImageTk, Image


# Function definitions
def red_func(rating):
    global ratings
    ratings[0] = rating
    red_title['text'] = 'Redness: ' + str(rating)
    # print('ratings: %d, %d, %d' % tuple(ratings))


def texture_func(rating):
    global ratings
    ratings[1] = rating
    texture_title['text'] = 'Texture: ' + str(rating)
    # print('ratings: %d, %d, %d' % tuple(ratings))


def even_func(rating):
    global ratings
    ratings[2] = rating
    even_title['text'] = 'Evenness: ' + str(rating)
    # print('ratings: %d, %d, %d' % tuple(ratings))


def submission():
    # Clear ratings
    global ratings
    global filename
    print(str(filename) + ': %d, %d, %d' % tuple(ratings))

    # Open new image
    if filenames:
        ratings = [0, 0, 0]
        red_title['text'] = 'Redness'
        texture_title['text'] = 'Texture'
        even_title['text'] = 'Evenness'
        filename = filenames.pop()
        img_new = Image.open(filename)
        img_new.thumbnail((w // 2, h // 2))
        img_new = ImageTk.PhotoImage(img_new)
        image['image'] = img_new
        image.image = img_new
        image.grid(row=1,
                   column=0,
                   columnspan=10)
    else:
        submit.destroy()
        finished = tk.Label(root,
                            text='Done!',
                            fg='black',
                            font='Helvetica 16')
        finished.grid(row=8,
                      column=0,
                      columnspan=10)


# Get filenames
filenames = sorted([x for x in os.listdir('./') if x[-3:] == 'jpg'],
                   reverse=True)

# Preallocate ratings
ratings = [0, 0, 0]

# Define root widget (parent of all subwidgets)
root = tk.Tk()

# Get screen pixel dimensions
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
# root.geometry("%dx%d+0+0" % (w // 2, h))

# Title
title = tk.Label(root,
                 text='Dermatology Image Labeler',
                 fg='black',
                 font='Helvetica 20 bold').grid(row=0,
                                                column=0,
                                                columnspan=10)

# Image
filename = filenames.pop()
img = Image.open(filename)
img.thumbnail((w // 2, h // 2))
img = ImageTk.PhotoImage(img)
image = tk.Label(root,
                 height=h // 2,
                 width=h // 2,
                 image=img)
image.grid(row=1,
           column=0,
           columnspan=10)

# Redness
red_title = tk.Label(root,
                     text='Redness',
                     fg='black',
                     font='Helvetica 16')
red_title.grid(row=2,
               column=0,
               columnspan=10)

for i in range(10):
    red_button = tk.Button(root, text=str(i),
                           command=lambda rating=i: red_func(rating))
    red_button.grid(row=3, column=i)

# Texture
texture_title = tk.Label(root,
                         text='Texture',
                         fg='black',
                         font='Helvetica 16')
texture_title.grid(row=4,
                   column=0,
                   columnspan=10)

for i in range(10):
    texture_button = tk.Button(root, text=str(i),
                               command=lambda rating=i: texture_func(rating))
    texture_button.grid(row=5, column=i)

# Evenness
even_title = tk.Label(root,
                      text='Evenness',
                      fg='black',
                      font='Helvetica 16')
even_title.grid(row=6,
                column=0,
                columnspan=10)

for i in range(10):
    even_button = tk.Button(root, text=str(i),
                            command=lambda rating=i: even_func(rating))
    even_button.grid(row=7, column=i)

# Submit button
submit = tk.Button(root,
                   text='Submit',
                   font='Helvetica 16 bold',
                   command=submission)
submit.grid(row=8,
            column=0,
            columnspan=10)

# Make GUI interface appear
root.mainloop()
