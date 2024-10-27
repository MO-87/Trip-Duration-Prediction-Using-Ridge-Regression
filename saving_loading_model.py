import pickle


def saving_model(model, saving_path):

    with open(saving_path, 'wb') as file:
        pickle.dump(model, file)


def loading_model(loading_path):

    with open(loading_path, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model
