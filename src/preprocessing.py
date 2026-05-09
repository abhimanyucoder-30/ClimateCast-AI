import pandas as pd

def load_and_clean_data(file_path, target_column="meantemp"):

    df = pd.read_csv(file_path)
    df.columns= df.columns.str.lower().str.strip()
    if "date" not in df.columns:
        raise ValueError("The dataset must contain a 'date' column.")
    
    if target_column not in df.columns:
        raise ValueError(f"The dataset must contain the target column '{target_column}'.")

    df= df[["date", target_column]]

    df= df.rename(columns={
        "date": "ds",
        target_column: "y"
    })

    df["ds"]= pd.to_datetime(df["ds"], errors="coerce")
    df["y"]= pd.to_numeric(df["y"], errors="coerce")

    df= df.dropna()  

    df=df.sort_values("ds")
    df= df.drop_duplicates(subset="ds", keep="first")
    df= df.reset_index(drop=True)

    return df 

def get_available_numeric_columns(file_path):

    df = pd.read_csv(file_path)
    df.columns= df.columns.str.lower().str.strip()
    numeric_columns= df.select_dtypes(include=["number"]).columns.tolist()

    if "date" in numeric_columns:
        numeric_columns.remove("date")

    return numeric_columns