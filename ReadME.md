## Welcome to HandCraft!

This project uses Convolutional Neural Networks to allow you to control Minecraft with just your hands. You can easily adapt the controls for other games.

## To run this:

1. Change the class on line 26 of DataCollection.py and run to get some training images. Currently the program uses classes "open" and "fist" for the left hand; "zero", "one", and "two" for the right hand. To collect data, run the program, stand wherever you want in frame, and hit "L" to take a picture, it will automatically detect your hand. I recommend 300-400 images for each class for good results!
2. Run TrainNNLeft.py and TrainNNRight.py to build 2 Keras models off of your training images. Both files assume your training images are in a folder called "TrainingImages"
3. Run main.py. It takes about 1 minute to boot up, then will start detecting your hands! Main.py has some tuneable parameters like the left hand movement boxes (line 27) and how fast the camera moves (line 39)
4. If you find the program is detecting the wrong hand class/position a lot, try recapturing more training images, or increasing VOTE_WINDOW on line 49 in main.py. This will average out predictions more so 1 wrong frame prediction does not mess you up!
