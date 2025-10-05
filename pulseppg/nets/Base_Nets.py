class Base_NetConfig:
    def __init__(
        self,
        # model parameters
        net_folder: str,
        net_file: str,
        # model parameters
        params: dict = {},
    ):
        self.net_folder = net_folder
        self.net_file = net_file

        self.params = params
