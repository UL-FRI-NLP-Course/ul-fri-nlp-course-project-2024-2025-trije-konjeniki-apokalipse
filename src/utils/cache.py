import numpy as np
import pandas as pd
import os



def _create_directory(file_path):
    """Creates the directory for the file path if it doesn't exist."""
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)


def csv(file_path, create_directory=True):
    """Stores the dataframe returned by a function into a CSV file. When the function is called again,
    it will return the dataframe from the CSV file instead of executing the function again."""

    def decorator(func):

        def wrapper(*args, **kwargs):

            if os.path.exists(file_path):
                print(f'Loading cached DataFrame from {file_path}')
                return pd.read_csv(file_path)
            
            result = func(*args, **kwargs)

            if create_directory: _create_directory(file_path)
            result.to_csv(file_path, index=False)
            return result
        
        return wrapper
    
    return decorator



def npz(file_path, create_directory=True):
    """Stores a single numpy array or a tuple of numpy arrays into a .npz file.
    When the function is called again, it will return the numpy array(s) from the
    .npz file instead of executing the function again."""

    def decorator(func):

        def wrapper(*args, **kwargs):

            if os.path.exists(file_path):
                print(f'Loading cached numpy arrays from {file_path}')
                result = np.load(file_path)
                result = tuple(result[key] for key in result.files)
                retval = result[0] if len(result) == 1 else result
                return retval
                
            
            result = func(*args, **kwargs)
            if not isinstance(result, tuple):
                result = (result,)

            if create_directory: _create_directory(file_path)
            np.savez(file_path, *result)
            
            retval = result[0] if len(result) == 1 else result
            return result
        
        return wrapper
    
    return decorator