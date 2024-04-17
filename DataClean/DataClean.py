# import pandas as pd

# def clean_data(input_file, output_file):
#     # Read the xls file into a DataFrame
#     df = pd.read_excel(input_file)

#     # Remove rows with weight = 0
#     df = df[df['Weight'] != 0]

#     # Group by 'DateId', 'OrderNo', 'ItemId'
#     grouped = df.groupby(['DateId', 'OrderNo', 'ItemId'])

#     # Create the action column
#     df['action'] = ''

#     # Get the index of the 'Weight' column
#     weight_index = df.columns.get_loc('Weight')

#     # Insert the 'action' column right next to the 'Weight' column
#     df.insert(weight_index + 1, 'action', '')

#     # Iterate over the groups
#     for _, group in grouped:
#         total_weight = group['Weight'].sum()

#         # Update the action column based on the total weight
#         if total_weight == 0:
#             df.loc[group.index, 'action'] = 'U'
#         elif total_weight < 0:
#             df.loc[group.index, 'action'] = 'C'
#         else:
#             df.loc[group.index, 'action'] = 'O'

#     # Save the cleaned data to a new csv file
#     df.to_csv(output_file, index=False)

# # Example usage
# input_file = 'Pappit.xlsx'
# output_file = 'cleaned_data.csv'
# clean_data(input_file, output_file)








import pandas as pd

def clean_data(file_path):
    # Read the xls file
    df = pd.read_excel(file_path)
    
    # Remove all rows with weight = 0
    df = df[df['Weight'] != 0]
    
    # Group by 'DateId', 'OrderNo', 'ItemId'
    grouped = df.groupby(['DateId', 'OrderNo', 'ItemId'])
    
    # Define a function to assign the action based on the total weight of the group
    def assign_action(group):
        total_weight = group['Weight'].sum()
        if total_weight == 0:
            action = 'U'  # Updated
        elif total_weight < 0:
            action = 'C'  # Cancelled
        else:
            action = 'O'  # Ordered
        group['Action'] = action
        return group
    
    # Apply the function to each group and combine the results
    df = grouped.apply(assign_action)
    
    # Reorder columns to place 'Action' right after 'Weight'
    # Make a list of all column names and insert 'Action' after 'Weight'
    cols = list(df.columns)
    weight_index = cols.index('Weight')
    # Move 'Action' to the right of 'Weight'
    cols.insert(weight_index + 1, cols.pop(cols.index('Action')))
    df = df[cols]
    
    # Write the cleaned data to a new CSV file
    df.to_csv('cleaned_data1.csv', index=False)

# Use the function with the file path you specify later
clean_data('Pappit.xlsx')
