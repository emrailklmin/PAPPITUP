import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel('Pappit.xlsx')

class Division:
    def __init__(self, data, division_name):
        self.data = data[data['DivisionName'] == division_name]
        self.division_name = division_name

    def summarize_orders_by_customer(self):
        summary = self.data.groupby('CustomerId').agg(
            Total_Quantity=('Quantity', 'sum'),
            Total_Weight=('Weight', 'sum')
        ).sort_values(by=['Total_Quantity', 'Total_Weight'], ascending=False)
        return summary


class Customer:
    def __init__(self, data, customer_id):
        self.data = data[data['CustomerId'] == customer_id]
        self.customer_id = customer_id

    def summarize_orders(self):
        summary = self.data.groupby(['CustomerId', 'DivisionName']).agg(
            Total_Quantity=('Quantity', 'sum'),
            Total_Weight=('Weight', 'sum')
        ).sort_values(by=['Total_Quantity', 'Total_Weight'], ascending=False)
        return summary


class Product:
    def __init__(self, data, product_category):
        self.data = data[data['ProductCategory'] == product_category]
        self.product_category = product_category
        # Här kan du lägga till metoder som är relevanta för en produktkategori


customer_instance = Customer(data, customer_id=12345)  # Byt ut 12345 mot riktigt CustomerId
customer_summary = customer_instance.summarize_orders()
print(customer_summary)

# Skapa en instans för en specifik division
division_East = Division(data, "PAPP East")  # Byt ut "PAPP East" mot riktigt divisionsnamn
division_east_summary = division_East.summarize_orders_by_customer()
print('east division summary biggest customers: \n', division_east_summary)

division_West = Division(data, "PAPP West")  # Byt ut "PAPP West" mot riktigt divisionsnamn
division_west_summary = division_West.summarize_orders_by_customer()
print('west division summary biggest customers: \n', division_west_summary)

division_North = Division(data, "PAPP North")  # Byt ut "PAPP North" mot riktigt divisionsnamn
division_north_summary = division_North.summarize_orders_by_customer()
print('north division summary biggest customers: \n', division_north_summary)

total_summary = data.groupby('CustomerId').agg(
    Total_Quantity=('Quantity', 'sum'),
    Total_Weight=('Weight', 'sum')
).sort_values(by=['Total_Quantity', 'Total_Weight'], ascending=False)

total_summary['Percentage_Quantity'] = total_summary['Total_Quantity'] / total_summary['Total_Quantity'].sum() * 100
total_summary['Percentage_Weight'] = total_summary['Total_Weight'] / total_summary['Total_Weight'].sum() * 100

print('total summary biggest customers by quantity and weight:')
print(total_summary)

# Anta att 'data' är din DataFrame som du redan har läst in från Excel-filen
total_summary = data.groupby('CustomerId').agg(
    Total_Quantity=('Quantity', 'sum'),
    Total_Weight=('Weight', 'sum')
).sort_values(by=['Total_Quantity', 'Total_Weight'], ascending=False)

total_summary['Percentage_Quantity'] = total_summary['Total_Quantity'] / total_summary['Total_Quantity'].sum() * 100
total_summary['Percentage_Weight'] = total_summary['Total_Weight'] / total_summary['Total_Weight'].sum() * 100

# Skriv ut tabellen till en HTML-fil (kan öppnas med en webbläsare)
total_summary.to_html('total_summary_table.html')
