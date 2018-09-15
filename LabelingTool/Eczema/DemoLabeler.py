# Import modules
import os
import tkinter as tk
from PIL import ImageTk, Image


# Function definitions
def decide(rating):
    global ratings

    # Assign user selected rating
    ratings[0] = rating

    # Make number glow blue to indicate selection
    decision[rating]['fg'] = 'blue'
    decision[rating - 1]['fg'] = 'black'


def options(rating):
    global ratings

    # Assign user selected rating
    ratings[rating + 1] = 0 if ratings[rating + 1] else 1

    # Make number glow blue to indicate selection
    issues[rating]['fg'] = 'blue' if issues[rating]['fg'] == 'black' else 'black'


def refresh():
    global ratings
    global filename

    # Reset decision buttons
    for x in range(2):
        decision[x]['fg'] = 'black'

    # Reset issue buttons
    for x in range(8):
        issues[x]['fg'] = 'black'
        ratings = [None]
        ratings += 8 * [0]

    # Load next image
    if file_paths:
        file_title['text'] = file_paths[-1][-40:] + ' (%d remaining)' % len(file_paths)
        filename = file_paths.pop()
        img_new = Image.open('DermImages/' + filename)
        img_new.thumbnail((w//2, h//2))
        img_new = ImageTk.PhotoImage(img_new)
        image['image'] = img_new
        image.image = img_new
        image.grid(row=2, column=0, columnspan=12)

    # Remove buttons when no more images left
    else:
        skip.destroy()
        submit.destroy()
        file_title['text'] = filename[-40:] + ' (0 remaining)'
        finished = tk.Label(root, text='Done!', fg='black', font='Helvetica 18 bold')
        finished.grid(row=7, column=0, columnspan=12)


def skip():
    global ratings
    global filename

    # Write "no rating" to file for skipped images
    if filename:
        with open('DermImages/ratings.txt', 'a') as file:
            file.write(9 * '_, ' + str(filename) + ', ' + notes_field.get() + '\n')
        notes_field.delete(0, 'end')
        print(str(filename) + ': no rating')

        # Reset buttons and load next image
        refresh()


def submit():
    global ratings
    global filename

    # Only write ratings to file when all ratings have been chosen
    if None not in ratings:
        with open('DermImages/ratings.txt', 'a') as file:
            file.write(9 * '%d, ' % tuple(ratings) + str(filename) + ', ' + notes_field.get() + '\n')
        print(str(filename) + ': ' + str(ratings))
        notes_field.delete(0, 'end')

        # Reset buttons and load next image
        refresh()


# Get filenames
print('Working directory: ' + os.getcwd())
file_paths = [y+'/'+x for y in os.listdir('DermImages') if (y[0] != '.') and (y != 'ratings.txt')
              for x in os.listdir('DermImages/'+y) if x[-4:] == '.jpg']

# Sort filenames and remove already labeled ones from list
file_paths = sorted(file_paths, reverse=True)
if os.path.isfile('DermImages/ratings.txt'):
    with open('DermImages/ratings.txt', 'r') as image_name:
        file_paths = file_paths[:file_paths.index(list(image_name)[-1].split(',')[-2][1:])]

# Preallocate ratings
ratings = [None]
ratings += 8 * [0]

# Define root widget (parent of all subwidgets)
root = tk.Tk()

# Get screen pixel dimensions
w, h = root.winfo_screenwidth(), root.winfo_screenheight()

# Title
title = tk.Label(root, text='Dermatology Image Labeler', fg='black', font='Helvetica 24 bold')
title.grid(row=0, column=0, columnspan=12)

# Display image
if file_paths:
    file_title = tk.Label(root, text=file_paths[-1][-40:] + ' (%d remaining)' % len(file_paths),
                          fg='black', font='Helvetica 18')
    file_title.grid(row=1, column=0, columnspan=12)
    filename = file_paths.pop()
    img = Image.open('DermImages/' + filename)
    img.thumbnail((w // 2, h // 2))
    img = ImageTk.PhotoImage(img)
    image = tk.Label(root, height=h // 2, width=h // 2, image=img)
    image.grid(row=2, column=0, columnspan=12)

# Display blank if no images left
else:
    file_title = tk.Label(root, text='(%d remaining)' % len(file_paths),
                          fg='black', font='Helvetica 18')
    file_title.grid(row=1, column=0, columnspan=12)
    img = Image.new('RGB', (1, 1), 'white')
    img.thumbnail((w // 2, h // 2))
    img = ImageTk.PhotoImage(img)
    image = tk.Label(root, height=h // 2, width=h // 2, image=img)
    image.grid(row=2, column=0, columnspan=12)

# Decision
decision = []
decision += [tk.Button(root, text='Reject', fg='black', font='Arial 20',
                       width=h // 80, command=lambda rating=0: decide(rating))]
decision += [tk.Button(root, text='Accept', fg='black', font='Arial 20',
                       width=h // 80, command=lambda rating=1: decide(rating))]
decision[0].grid(row=3, column=5, columnspan=3)
decision[1].grid(row=3, column=2, columnspan=3)

# Issues
issues = []
categories = ['Too Bright', 'Too Dim', 'Too Close', 'Too Far',
              'Face Obscured', 'Out of Focus', 'Looking Away', 'Other']
for i, j in enumerate(categories):
    issues += [tk.Button(root, text=j, width=h//80, fg='black', font='Arial 20',
                         command=lambda rating=i: options(rating))]
    issues[i].grid(row=4 + (i // 4), column=3 * (i % 4), columnspan=2)

# Gap
gap = tk.Label(root, bg='white')
gap.grid(row=6, column=0, columnspan=12)

# Notes
notes_title = tk.Label(root, text='Notes', fg='black', font='Arial 20')
notes_title.grid(row=7, column=0, columnspan=12)

# Skip button
skip = tk.Button(root, text='Skip', font='Arial 20', width=h//80, command=skip)
skip.grid(row=8, column=0, columnspan=2)

# Submit button
submit = tk.Button(root, text='Submit', font='Arial 20', width=h//80, command=submit)
submit.grid(row=8, column=9, columnspan=2)

# Tag form field
notes_field = tk.Entry(root)
notes_field.grid(row=8, column=2, columnspan=6)

# Make GUI interface appear
root.mainloop()
