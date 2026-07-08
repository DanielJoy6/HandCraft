## Welcome to HandCraft!

This project uses Convolutional Neural Networks to allow you to control Minecraft with just your hands and a webcam. You can easily adapt the controls for other games. HandCraft uses MediapIpe Hands for real-time hand tracking, then 2 trained CNN classifiers built on top of MobileNetV2 to translate hand gestures into keyboard/mouse inputs.

The left hand controls movement (4 zones for WSAD) as well as opening and closing inventory (close/open hand)
The right hand controls camera movement (just moving hand physically around) and the mouse left/right click (hold 1 or 2 fingers up).

## To run this yourself:

1. Change the class on line 26 of DataCollection.py and run to get some training images. Currently the program uses classes "open" and "fist" for the left hand; "zero", "one", and "two" for the right hand. To collect data, run the program, stand wherever you want in frame, and hit "L" to take a picture, it will automatically detect your hand.

- I recommend 300-400 images for each class for good results!
- Try to wear clothing that won't get in the way of hand detection, either a short-sleeve shirt or wear sleeves that contrast to your skin tone.
- Try to slightly vary your hand angle, distance, lighting, etc so the model gets a good understanding of how the class can look.
- You can hit 'Q' to quit the program anytime and move on to the next class, remember to change line 26 to the next class.

2. Run TrainNNLeft.py and TrainNNRight.py to build 2 Keras models off of your training images. Both files assume your training images are in a folder called "TrainingImages"
3. Run main.py. It takes about 1 minute to boot up, then will start detecting your hands! Main.py has some tuneable parameters like the left hand movement boxes (line 27) and how fast the camera moves (line 39)
4. If you find the program is detecting the wrong hand class/position a lot, try recapturing more training images, or increasing VOTE_WINDOW on line 49 in main.py. This will average out predictions more so 1 wrong frame prediction does not mess you up!

## Requirements
- Python 3.9+
- Webcam
- Python packages: opencv, mediapipe, keras, tensorflow, pynput, scikit-learn, pillow, numpy

## How can I adapt this to other games?
1. Make sure you aren't going to get banned for cheating first, since this is a program controlling your movement
2. Change the MOVEMENT_ZONES, key mappings in update_movement(), and the functions send_inventory_command() and send_click_command() in main.py to whatever controls you want.

FPS on my laptop was about 16 for the camera, but I could reach 50 FPS in Minecraft itself.

You are welcome to use this project however you want, please just credit this page if you do.
