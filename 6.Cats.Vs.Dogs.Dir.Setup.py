import os, shutil

original_data_dir = '/home/andrew/PycharmProjects/AndrewProjects/CatDog/CatDogData/'
working_dir = '/home/andrew/PycharmProjects/AndrewProjects/CatDog/CatDogWorking'

train_dir = os.path.join(working_dir, 'train')
os.mkdir(train_dir)
val_dir = os.path.join(working_dir, 'val')
os.mkdir(val_dir)
test_dir = os.path.join(working_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

val_cats_dir = os.path.join(val_dir, 'cats')
os.mkdir(val_cats_dir)
val_dogs_dir = os.path.join(val_dir, 'dogs')
os.mkdir(val_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(5000)]
for fname in fnames:
    src = os.path.join(original_data_dir, 'train', fname)
    dest = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dest)

fnames = ['cat.{}.jpg'.format(i) for i in range(5000, 7000)]
for fname in fnames:
    src = os.path.join(original_data_dir, 'train', fname)
    dest = os.path.join(val_cats_dir, fname)
    shutil.copyfile(src, dest)

fnames = ['cat.{}.jpg'.format(i) for i in range(7000,10000)]
for fname in fnames:
    src = os.path.join(original_data_dir, 'train', fname)
    dest = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dest)

fnames = ['dog.{}.jpg'.format(i) for i in range(5000)]
for fname in fnames:
    src = os.path.join(original_data_dir, 'train', fname)
    dest = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dest)

fnames = ['dog.{}.jpg'.format(i) for i in range(5000, 7000)]
for fname in fnames:
    src = os.path.join(original_data_dir, 'train', fname)
    dest = os.path.join(val_dogs_dir, fname)
    shutil.copyfile(src, dest)

fnames = ['dog.{}.jpg'.format(i) for i in range(7000, 10000)]
for fname in fnames:
    src = os.path.join(original_data_dir, 'train', fname)
    dest = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dest)
