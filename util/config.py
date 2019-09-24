import json


def GetConfig():
    with open('./config/config.json') as json_file:
        return json.load(json_file)
