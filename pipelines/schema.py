class Schema:
    
    def __init__(self, schema_dict):
        self.schema_dict = schema_dict

    def columns(self):
        return list(map(lambda column: column['name'], self.schema_dict))
    
    def inputs(self):
        return self.columns()[:-1]

    def target(self):
        for feature_dict in self.schema_dict:
            if feature_dict['type'] == 'target':
                return feature_dict['name']
        raise ValueError('Target variable not found.')
    
            
