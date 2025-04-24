




from decimal import Decimal
import json
import pathlib
import pandas as pd

df = pd.read_csv(f"{pathlib.Path().absolute()}/employee_data.csv")

print(df.shape)

json_records = df.to_json(orient='records')

data = json.loads(json_records)

output_path = pathlib.Path().absolute() / "output.json"
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

