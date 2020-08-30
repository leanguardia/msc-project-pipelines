class RegressionPredictor():
    def __init__(self, model):
        self.model = model

    def predict(self, X, decimals=2):
        """ Performs one Prediction.
        
        Parameters:
        - X: Input array compatible with model interface for inference.
        - decimals (int): Number of decimals required

        Returns: (int) Prediction result. 
        """

        prediction = self.model.predict(X)[0]
        if prediction < 0: prediction = 0
        return [round(prediction, decimals)]
