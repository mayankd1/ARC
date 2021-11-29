#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

def solve_aabf363d(x):
    # Get the shape of i/p NP Array
    row,col=x.shape
    # Store the Color value from the first column-last row 
    colorEnd = x[row-1,0] 
    # Changing the Color of the first column-last row to black
    x[row-1,0] = 0
    # Changing the Color of the figure by checking all non-zero values in the matrix   
    x[x > 0] = colorEnd
    # Return O/p
    return x

# Solution for Training example c8cbb738.json
def solve_c8cbb738(x):
    data = np.array(x)
    # Storing Colors and their occurences
    uniq, counts = np.unique(data, return_counts=True)
    # Check the indexes for background color
    index = 0
    for i in counts:
        if i == max(counts):
            break
        index += 1
    # Storing background color
    backG = uniq[index]
    # Deleting the Color code and counts to create new lists with primary colors
    uniq = np.delete(uniq, index)
    counts = np.delete(counts, index)

    # Declaring variables to store (key,value) of Color(as keys) and Coordinates(as values) 
    listOfCoordinates={}
    # Dimension of O/p array
    dim_out_arr = []
    # Lambda Fn to extract x,y coordinates
    c = lambda x,y: [list(c) for c in zip(x, y)]
    # Itering over primary colors to get the coordinates for each
    for i in range(len(uniq)):
        a = np.where(data == uniq[i])
        listOfCoordinates[uniq[i]] = c(a[0],a[1])
        dim_out_arr.append(max(a[0]) - min(a[0]))
        dim_out_arr.append(max(a[1]) - min(a[1]))
    # Using max Fn to get the dimensions of O/p array
    dim_out = max(dim_out_arr)

    # Method to slice and traverse the Color matrix and determining it's position on O/p matrix
    def _sliceCoordinates(val, dim):
        d=np.array(val)
        # Checking minimum coordinates for x and y axis
        minx=min(d.T[0])
        miny=min(d.T[1])
        d=d-[minx,miny]
        # Checking the Color axis if it contains the dimension of O/p array
        # To determine if the shape of Color grid will fit to O/p  array or requires a shift in X or Y axis
        if dim in d.T[0]:
            if dim in d.T[1]:
                pass
        # If Color grid does not fit in Y-axis, make a shift to match the O/p
            else:
                d.T[1]=d.T[1]+(dim-max(d.T[1]))/2
        # If Color grid does not fit in X-axis, make a shift to match the O/p
        else:
            d.T[0]=d.T[0]+(dim-max(d.T[0]))/2
        # Return the index for each color wrt output
        return d

    # Declaring a dict to store Color(as Keys) and final indexes(as Values)
    newCoordinates={}
    # Creating an output matrix, filled with the background color
    output_matrix = np.full((dim_out+1, dim_out+1), backG)
    # Iterating over each key to place the colors at the desired x,y coordinates
    for key in listOfCoordinates.keys():
        for n_coor in _sliceCoordinates(listOfCoordinates[key], dim_out):
            output_matrix[tuple(n_coor)] = key

    return output_matrix

# def solve_05269061(x):
#     return x


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

