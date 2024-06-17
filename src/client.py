import requests
import pandas as pd 
import ast
url = "http://127.0.0.1:8000/linear_predict"

data = pd.read_csv('../data/articles.csv')
data["meshroot"] = data["meshroot"].apply(ast.literal_eval)  # apply literal_eval because lists appear as strings

 
text = data["abstractText"][7000]

payload = {
    "text": text
}

# Make request to the endpoint 
response = requests.post(url, json=payload)

print(response.json())
