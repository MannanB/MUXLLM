class ModelNotAvailable(Exception):
    pass

class CloudProvider:
    def __init__(self, available_models : list[str], model_alias : dict[str, str]):
        self.available_models = available_models
        self.model_alias = model_alias
        self.client = None
        self.async_client = None

    def validate_model(self, model : str): 
        if model in self.model_alias:
            model = self.model_alias[model]

        # if available_models is empty, all models are availablel;mokmp;kmp;okpokp
        if len(self.available_models) > 0 and not model in self.available_models:
            raise ModelNotAvailable(f"Model {model} is not available. Available models are {', '.join(self.available_models)}")
        
        return model

    def get_response(self, messages : list[dict[str, str]], model : str, **kwargs):
        model = self.validate_model(model)
        
        response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs) 
        
        return response
    
    async def get_response_async(self, messages : list[dict[str, str]], model : str, **kwargs):
        model = self.validate_model(model)

        response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs) 
        
        return response
    
    def get_response_stream(self, messages : list[dict[str, str]], model : str, **kwargs):
        model = self.validate_model(model)
        
        response = self.client.chat.completions.create_stream(
                    model=model,
                    messages=messages,
                    stream=True,
                    **kwargs) 
        
        return response