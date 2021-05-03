import json
import pickle
import numpy as np

# global variables
__locations = None
__data_columns = None
__model = None

#function to predict price
def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if 0 <= loc_index:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:] #since first 3columns are sqft, bath, bhk

    with open("./artifacts/bangalore_home_price_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")

#function to return all location names
def get_location_names():
    return __locations

#function to return all location names
def get_data_columns():
    return __data_columns

# this will import artifacts like columns
if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())

    #location,sqft,bhk,bath
    print(get_estimated_price("1st Phase JP Nagar", 1000, 3, 3))
    print(get_estimated_price("1st Phase JP Nagar", 1000, 2, 2))
    print(get_estimated_price("Kalhalli", 1000, 2, 2))
    print(get_estimated_price("Ejjipura", 1000, 2, 2))
