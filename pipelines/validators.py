class Validator():
    def __init__(self, column):
        self.column = column

    def validate(self, df):
        pass

class RangeValidator(Validator):
    def __init__(self, column, mini, maxi):
        super().__init__(column)
        self.mini = mini
        self.maxi = maxi
    
    def validate(self, df):
        is_out_of_range = lambda val: val < self.mini or val > self.maxi
        if df[self.column].apply(is_out_of_range).any():
            raise ValueError(f"'{self.column}' out of range")

class CategoryValidator(Validator):
    def __init__(self, column, categories):
        super().__init__(column)
        self.categories = categories
    
    def validate(self, df):
        if not (df[self.column].isin(self.categories)).all():
            raise ValueError(f"Invalid '{self.column}'")

