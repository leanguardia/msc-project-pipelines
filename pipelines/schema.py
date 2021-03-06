import pandas as pd
import numpy as np

from pipelines.validators import RangeValidator, CategoryValidator, NonNegativeValidator, PositiveValidator

class Schema:
    def __init__(self, features_list):
        self.features_list = features_list

    def features(self, which='all'):
        if which == 'input':
            return [feature['name'] for feature in self.features_list if self._is_input(feature)]
        if which == 'features':
            return [feature['name'] for feature in self.features_list if not self._is_target(feature)]
        return [feature['name'] for feature in self.features_list]

    def inputs(self):
        return self.features(which='input')

    def n_inputs(self):
        return len(self.inputs())

    def target(self):
        for feature_dict in self.features_list:
            if self._is_target(feature_dict):
                return feature_dict['name']
        raise ValueError('Target variable not found.')

    def dtypes(self, which='all'):
        if which == 'input':
            return [feature['dtype'] for feature in self.features_list if self._is_input(feature)]
        return [feature['dtype'] for feature in self.features_list]

    def validators(self, which='input'):
        validators = []
        if which=='input': 
            filtered_list = [feature for feature in self.features_list if self._is_input(feature)]
        elif which=='engineered':
            filtered_list = [feature for feature in self.features_list if not self._is_input(feature)]
        else: 
            filtered_list = self.features_list
    
        for feature in filtered_list:
            if 'range' in feature:
                validators.append(self._build_range_validator(feature))
            if 'categories' in feature:
                validators.append(self._build_category_validator(feature))
            if 'non_negative' in feature:
                validators.append(self._build_non_negative_validator(feature))
            if 'positive' in feature:
                validators.append(self._build_positive_validator(feature))
        return validators

    def _is_input(self, feature):
        return ('type' in feature) and (feature['type'] == 'input' or feature['type'] == 'target')

    def _is_target(self, feature):
        return ('type' in feature) and (feature['type'] == 'target')

    def _build_range_validator(self, feature):
        mini, maxi = feature['range']
        return RangeValidator(feature['name'], mini, maxi)

    def _build_category_validator(self, feature):
        return CategoryValidator(feature['name'], feature['categories'])

    def _build_non_negative_validator(self, feature):
        return NonNegativeValidator(feature['name'])

    def _build_positive_validator(self, feature):
        return PositiveValidator(feature['name'])
