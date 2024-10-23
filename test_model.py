from mainV2 import *

parameters = load_saved_params()

run = True
while run: 
    index = int(input("Enter an image number: "))

    if index < 0: 
        run = False 
        print("Exiting now...")
        break

    test_prediction(index, parameters)

