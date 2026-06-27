import torch


class BaseModel(torch.nn.Module):
    def load(self, path, n_clusters=150):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        if 'pth.tar' in path:
            parameters = parameters["state_dict"]
            for k in list(parameters.keys()):
                if k.startswith('module.') :
                    parameters[k[len("module."):]] = parameters[k]
                del parameters[k]
             

        if 'dpt_hybrid-ade20k-53898607.pt' in path: # pas elegant, transitoire,

            if n_clusters==150:
                self.load_state_dict(parameters, strict = False)
            else:
                parameters.pop('scratch.output_conv.4.weight')
                parameters.pop('scratch.output_conv.4.bias')
                parameters.pop('auxlayer.4.weight')
                parameters.pop('auxlayer.4.bias')
                self.load_state_dict(parameters, strict = False)


        elif 'dpt_large-ade20k-b12dca68.pt' in path: # pas elegant, transitoire, 
            if n_clusters==150:
                self.load_state_dict(parameters, strict = False)
            else:
                parameters.pop('scratch.output_conv.4.weight')
                parameters.pop('scratch.output_conv.4.bias')
                parameters.pop('auxlayer.4.weight')
                parameters.pop('auxlayer.4.bias')
                self.load_state_dict(parameters, strict = False)

        else:
            self.load_state_dict(parameters)


