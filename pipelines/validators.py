class ValidationsRunner():
    def __init__(self):
        self.validators = []

    def add_validators(self, validators):
        is_validator = lambda obj: isinstance(obj, Validator)
        if not (isinstance(validators, list) and all(map(is_validator, validators))):
            raise TypeError('Parameter should be list of Validators')
    
        self.validators = self.validators + validators

    def validate(self, df):
        for validator in self.validators:
            validator.validate(df)

    def _is_validator(self, obj):
        return isinstance(obj, Validator)

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

class NonNegativeValidator(Validator):
    def __init__(self, column):
        super().__init__(column)
    
    def validate(self, df):
        if not (df[self.column] >= 0).all():
            raise ValueError(f"'{self.column}' should be non negative")

class PositiveValidator(Validator):
    def __init__(self, column):
        super().__init__(column)
    
    def validate(self, df):
        if not (df[self.column] > 0).all():
            raise ValueError(f"'{self.column}' should be positive")
