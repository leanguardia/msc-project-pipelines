class Schema:
    
    def __init__(self, schema_dict):
        self.schema_dict = schema_dict

    def columns(self):
        return list(self.schema_dict.keys())
    
    def inputs(self):
        return list(self.schema_dict.keys())[:-1]
