# Import modules
import os
import tkinter as tk
from PIL import ImageTk, Image


# Function definitions
def area(rating):
    global ratings

    # Assign user selected rating
    ratings[0] = rating

    # Make number glow blue to indicate selection
    for x in range(10):
        area_button[x]['fg'] = 'black'
        area_button[x]['font'] = 'Arial 18'
    area_button[rating]['fg'] = 'blue'
    area_button[rating]['font'] = 'Arial 18'


def redness(rating):
    global ratings

    # Assign user selected rating
    ratings[1] = rating

    # Make number glow blue to indicate selection
    for x in range(10):
        red_button[x]['fg'] = 'black'
        red_button[x]['font'] = 'Arial 18'
    red_button[rating]['fg'] = 'blue'
    red_button[rating]['font'] = 'Arial 18'


def desquamation(rating):
    global ratings

    # Assign user selected rating
    ratings[2] = rating

    # Make number glow blue to indicate selection
    for x in range(10):
        desq_button[x]['fg'] = 'black'
        desq_button[x]['font'] = 'Arial 18'
    desq_button[rating]['fg'] = 'blue'
    desq_button[rating]['font'] = 'Arial 18'


def refresh():
    global ratings
    global filename

    # Reset rating buttons
    for x in range(10):
        area_button[x]['fg'] = 'black'
        area_button[x]['font'] = 'Arial 18'
        red_button[x]['fg'] = 'black'
        red_button[x]['font'] = 'Arial 18'
        desq_button[x]['fg'] = 'black'
        desq_button[x]['font'] = 'Arial 18'
        ratings = [None, None, None]

    # Load next image
    if file_paths:
        file_title['text'] = file_paths[-1][-40:] + ' (%d remaining)' % len(file_paths)
        filename = file_paths.pop()
        img_new = Image.open('DermImages/' + filename)
        img_new.thumbnail((w//2, h//2))
        img_new = ImageTk.PhotoImage(img_new)
        image['image'] = img_new
        image.image = img_new
        image.grid(row=3, column=0, columnspan=10)

    # Remove buttons when no more images left
    else:
        skip.destroy()
        submit.destroy()
        file_title['text'] = filename[-40:] + ' (0 remaining)'
        finished = tk.Label(root, text='Done!', fg='black', font='Helvetica 18 bold')
        finished.grid(row=11, column=0, columnspan=10)


def skip():
    global ratings
    global filename

    # Write "no rating" to file for skipped images
    if filename:
        with open('DermImages/ratings.txt', 'a') as file:
            file.write('_, _, _, ' + str(filename) + ', ' +
                       username_form.get() + ', ' + notes_field.get() + '\n')
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
            file.write('%d, %d, %d, ' % tuple(ratings) + str(filename) +
                       ', ' + username_form.get() + ', ' + notes_field.get() + '\n')
        print(str(filename) + ': %d, %d, %d' % tuple(ratings))
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
        file_paths = file_paths[:file_paths.index(list(image_name)[-1].split(',')[3][1:])]

# Preallocate ratings
ratings = [None, None, None]

# Define root widget (parent of all subwidgets)
root = tk.Tk()

# Get screen pixel dimensions
w, h = root.winfo_screenwidth(), root.winfo_screenheight()

# Title
title = tk.Label(root, text='PASI Score Image Labeler', fg='black', font='Helvetica 24 bold')
title.grid(row=0, column=0, columnspan=10)

# Username
username = tk.Label(root, text='Name:', fg='black', font='Helvetica 18 bold')
username.grid(row=1, column=1, columnspan=3)
username_form = tk.Entry(root)
username_form.grid(row=1, column=3, columnspan=4)

# Display image
if file_paths:
    file_title = tk.Label(root, text=file_paths[-1][-40:] + ' (%d remaining)' % len(file_paths),
                          fg='black', font='Helvetica 18')
    file_title.grid(row=2, column=0, columnspan=10)
    filename = file_paths.pop()
    img = Image.open('DermImages/' + filename)
    img.thumbnail((w // 2, h // 2))
    img = ImageTk.PhotoImage(img)
    image = tk.Label(root, height=h // 2, width=h // 2, image=img)
    image.grid(row=3, column=0, columnspan=10)

# Display blank if no images left
else:
    file_title = tk.Label(root, text='(%d remaining)' % len(file_paths),
                          fg='black', font='Helvetica 18')
    file_title.grid(row=2, column=0, columnspan=10)
    img = Image.new('RGB', (1, 1), 'white')
    img.thumbnail((w // 2, h // 2))
    img = ImageTk.PhotoImage(img)
    image = tk.Label(root, height=h // 2, width=h // 2, image=img)
    image.grid(row=3, column=0, columnspan=10)

# Area
area_title = tk.Label(root, text='Area', fg='black', font='Helvetica 16 bold')
area_title.grid(row=4, column=0, columnspan=10)
area_button = []
for i in range(10):
    area_button += [tk.Button(root, text=str(i), width=h//250, fg='black', font='Arial 18',
                             command=lambda rating=i: area(rating))]
    area_button[i].grid(row=5, column=i)

# Induration
red_title = tk.Label(root, text='Induration', fg='black', font='Helvetica 16 bold')
red_title.grid(row=6, column=0, columnspan=10)
red_button = []
for i in range(10):
    red_button += [tk.Button(root, text=str(i), width=h//250, fg='black', font='Arial 18',
                                 command=lambda rating=i: redness(rating))]
    red_button[i].grid(row=7, column=i)

# Desquamation
desquamation_title = tk.Label(root, text='Desquamation', fg='black', font='Helvetica 16 bold')
desquamation_title.grid(row=8, column=0, columnspan=10)
desq_button = []
for i in range(10):
    desq_button += [tk.Button(root, text=str(i), width=h//250, fg='black', font='Arial 18',
                              command=lambda rating=i: desquamation(rating))]
    desq_button[i].grid(row=9, column=i)

# Gap
gap = tk.Label(root, bg='white')
gap.grid(row=10, column=0, columnspan=10)

# Notes
notes_title = tk.Label(root, text='Notes', fg='black', font='Helvetica 16 bold')
notes_title.grid(row=11, column=2, columnspan=6)

# Skip button
skip = tk.Button(root, text='Skip', font='Helvetica 16 bold', command=skip)
skip.grid(row=12, column=0, columnspan=2)

# Submit button
submit = tk.Button(root, text='Submit', font='Helvetica 16 bold', command=submit)
submit.grid(row=12, column=8, columnspan=2)

# Tag form field
notes_field = tk.Entry(root)
notes_field.grid(row=12, column=2, columnspan=6)

# Make GUI interface appear
root.mainloop()
