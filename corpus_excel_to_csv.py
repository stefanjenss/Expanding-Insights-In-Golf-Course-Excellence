"""
This script loads in an Excel file containing golf course reviews and converts the data into a CSV file.

The input file is named "top_and_non_golf_course_reviews.xlsx" and is expected to be located in the same directory as this script.

The output file is named "top_and_non_golf_course_reviews.csv" and is also saved in the same directory as this script.

The script uses the pandas library to read in the Excel file and convert it into a pandas DataFrame. The DataFrame is then converted into a CSV file using the to_csv() method.

"""

# Import the necessary libraries
import pandas as pd

# Load in the dataset
file_path = "top_and_non_golf_course_reviews.xlsx"
df = pd.read_excel(file_path)

# Convert the dataframe to a csv file
csv_name = "top_and_non_golf_course_reviews.csv"
df.to_csv(csv_name, index=False)