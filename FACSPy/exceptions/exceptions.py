class PanelDataTypeError(Exception):
    
    def __init__(self,
                 data_type):
        self.data_type = data_type
        self.message = f"Please provide the panel as a pandas dataframe, it was {self.data_type}"
        super().__init__(self.message)

class PanelFileTypeError(Exception):
    
    def __init__(self,
                 data_type):
        self.data_type = data_type
        self.message = f"Please provide the panel file name as a string, it was {self.data_type}. Example: panel_from_file = 'panel.txt'"
        super().__init__(self.message)

class PanelFileNotFoundError(Exception):

    def __init__(self,
                 panel_file):
        self.message = f"{panel_file} could not be found in the specified input directory!"
        super().__init__(self.message)

class PanelNoInputDirectoryError(Exception):

    def __init__(self):
        self.message = "No Input Directory has been provided."
        super().__init__(self.message)

class PanelCreationError(Exception):

    def __init__(self):
        self.message = "Panel could not be created because neither a file or a table was supplied and no flag to infer from data was created"
        super().__init__(self.message)


class WrongFACSPyEstimatorError(Exception):

    def __init__(self, 
                 message):
        self.message = message
        super().__init__(self.message)

class WrongFACSPyHyperparameterMethod(Exception):

    def __init__(self, 
                 message):
        self.message = message
        super().__init__(self.message)

class WrongFACSPyHyperparameterTuningDepth(Exception):

    def __init__(self, 
                 message):
        self.message = message
        super().__init__(self.message)



class HyperparameterTuningCustomError(Exception):

    def __init__(self, 
                 message):
        self.message = message
        super().__init__(self.message)


# class PanelCreationWarning(Warning):

#     def __init__(self,
#                  user_input: str) -> None:
#         self.message = f"Input parameters suggested to {user_input}, but panel was provided. The provided panel will be used."
#         super().__init__(self.message)
    
#     def __str__(self):
#         return repr(self.message)