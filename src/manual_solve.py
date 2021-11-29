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
    row,col=x.shape
     
    colorEnd = x[row-1,0] 
    
    x[row-1,0] = 0   
    x[x > 0] = colorEnd
    return x

def solve_c8cbb738(x):
    data = np.array(x)
    
    uniq, counts = np.unique(data, return_counts=True)

    index = 0
    for i in counts:
        if i == max(counts):
            break
        index += 1
    backG = uniq[index]
    uniq = np.delete(uniq, index)
    counts = np.delete(counts, index)

    listOfCoordinates={}
    dim_out_arr = []
    c = lambda x,y: [list(c) for c in zip(x, y)]

    for i in range(len(uniq)):
        a = np.where(data == uniq[i])
        listOfCoordinates[uniq[i]] = c(a[0],a[1])
        dim_out_arr.append(max(a[0]) - min(a[0]))
        dim_out_arr.append(max(a[1]) - min(a[1]))

    dim_out = max(dim_out_arr)

    def _sliceCoordinates(val, dim):
        d=np.array(val)
        #print(d)
        minx=min(d.T[0])
        miny=min(d.T[1])
        d=d-[minx,miny]

        if dim in d.T[0]:
            if dim in d.T[1]:
                pass
            else:
                d.T[1]=d.T[1]+(dim-max(d.T[1]))/2
        else:
            d.T[0]=d.T[0]+(dim-max(d.T[0]))/2
        return d

    newCoordinates={}
    output_matrix = np.full((dim_out+1, dim_out+1), backG)
    for key in listOfCoordinates.keys():
        for n_coor in _sliceCoordinates(listOfCoordinates[key], dim_out):
            output_matrix[tuple(n_coor)] = key

    return output_matrix

def solve_05269061(x):
    return x


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

