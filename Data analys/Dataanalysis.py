import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class Data:
    def __init__(self, file_path):
        self.file_path = file_path
        self.order_entry = None
        self.order_entry_np = None  # NumPy-kopia av order_entry
        self.backlog = None
        self.backlog_np = None  # NumPy-kopia av backlog

    def load_data(self):
        # Läs in 'OrderEntry'-fliken
        self.order_entry = pd.read_excel(self.file_path, sheet_name='OrderEntry')
        # Skapar NumPy-kopia
        self.order_entry_np = self.order_entry.to_numpy()

        # Läs in 'Backlog'-fliken
        self.backlog = pd.read_excel(self.file_path, sheet_name='Backlog')
        # Skapar NumPy-kopia
        self.backlog_np = self.backlog.to_numpy()

        # Returnerar både DataFrames och NumPy-arrays
        return (self.order_entry, self.order_entry_np), (self.backlog, self.backlog_np)

def visualize_pivot_table(dataframe, index, columns, values, aggfunc):
    """
    Skapar och visualiserar en pivottabell.

    :param dataframe: DataFrame att använda.
    :param index: Kolumnnamn att använda som index i pivottabellen.
    :param columns: Kolumnnamn att använda som kolumner i pivottabellen.
    :param values: Kolumnnamn vars värden ska aggregeras.
    :param aggfunc: Aggregeringsfunktion att använda på 'values'.
    """
    
    # Skapar pivottabellen
    pivot_table = pd.pivot_table(dataframe, index=index, columns=columns, values=values, aggfunc=aggfunc)
    
    # Visualiserar pivottabellen som en heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
    plt.title(f"Heatmap av {aggfunc} för '{values}' över '{index}' och '{columns}'")
    plt.ylabel(index)
    plt.xlabel(columns)
    plt.show()

def visualize_correlation(dataframe):
    """
    Beräknar och visualiserar korrelationsmatrisen för numeriska variabler i en DataFrame.

    :param dataframe: DataFrame att analysera.
    """
    
    # Filtrera DataFrame till endast numeriska kolumner
    numeric_df = dataframe.select_dtypes(include=[np.number])

    # Beräkna korrelationsmatrisen för numeriska kolumner
    correlation_matrix = numeric_df.corr()

    # Visualisera korrelationsmatrisen som en heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Korrelationsmatris')
    plt.show()

def visualize_weight_over_time(dataframe, time_frame='M'):
    """
    Visualiserar total vikt och vikt per division över tid, aggregerat på veckor eller månader.

    :param dataframe: DataFrame som innehåller kolumnerna 'Date', 'Weight' och 'DivisionName'.
    :param time_frame: Tidsram för aggregering; 'W' för veckor, 'M' för månader.
    """
    
    # Kontrollera att 'Date' är i datetime-format
    dataframe['DateId'] = pd.to_datetime(dataframe['DateId'])

    # Sätt 'Date' som index
    dataframe.set_index('DateId', inplace=True)

    # Aggregera total vikt över tid baserat på vald tidsram
    total_weight = dataframe['Weight'].resample(time_frame).sum()

    # Återställ index för att 'Date' ska vara en kolumn igen
    total_weight = total_weight.reset_index()

    # Aggregera vikt per division över tid baserat på vald tidsram
    weight_per_division = dataframe.groupby('DivisionName')['Weight'].resample(time_frame).sum()
    weight_per_division = weight_per_division.reset_index()

    # Skapar linjediagram för total vikt över tid
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)  # 1 rad, 2 kolumner, första subplot
    plt.plot(total_weight['DateId'], total_weight['Weight'], marker='o', linestyle='-')
    plt.title('Total Vikt Över Tid')
    plt.xlabel('Datum')
    plt.ylabel('Total Vikt')
    plt.xticks(rotation=45)

    # Skapar linjediagram för vikt per division över tid
    plt.subplot(1, 2, 2)  # 1 rad, 2 kolumner, andra subplot
    sns.lineplot(data=weight_per_division, x='DateId', y='Weight', hue='DivisionName', marker='o')
    plt.title('Vikt per Division Över Tid')
    plt.xlabel('Datum')
    plt.ylabel('Vikt')
    plt.legend(title='Division', loc='upper left')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# Använda funktionen
# visualize_weight_over_time(ditt_dataframe, 'W')  # För veckovis aggregering
# visualize_weight_over_time(ditt_dataframe, 'M')  # För månatlig aggregering
    
def perform_holt_winters_analysis(dataframe, numeric_column, seasonal_periods, periodicity):
    """
    Utför Holt-Winters analys på en tidsserie efter att ha aggregerat datan till månader.

    :param dataframe: DataFrame som innehåller tidsseriedatan.
    :param numeric_column: Namnet på den numeriska kolumnen för analys ('Weight' eller 'Quantity').
    :param seasonal_periods: Antal observationer per säsong (t.ex. 12 för månadsdata med årlig säsongsvariation).
    """
    
    # Kontrollera att indexet är i datetime-format och sätt det som index
    dataframe['DateId'] = pd.to_datetime(dataframe['DateId'])
    dataframe.set_index('DateId', inplace=True)
    
    # Aggregera datan till månader
    monthly_data = dataframe[numeric_column].resample(periodicity).sum()


    
    # Skapa modellinstansen med manuellt specificerade utjämningskonstanter
    model = ExponentialSmoothing(
        monthly_data, 
        trend='add', 
        seasonal='add', 
    )

    # seasonal_periods=52,
    # smoothing_level=0.2,  # Exempelvärde för alpha
    # smoothing_slope=0.05,  # Exempelvärde för beta
    # smoothing_seasonal=0.2,  # Exempelvärde för gamma
    # optimized=False  # Förhindrar automatisk optimering av parametrar

    # Anpassa modellen
    fitted_model = model.fit()

    # Gör prognoser
    forecast = fitted_model.forecast(steps=seasonal_periods)

    print("Utjämningskonstanter:")
    print(f"Alpha (Nivå): {fitted_model.params.get('smoothing_level', 'Ej tillämplig'):.4f}")

    # Använder .get() för att undvika KeyError om 'smoothing_slope' inte finns
    beta = fitted_model.params.get('smoothing_slope', None)
    print(f"Beta (Trend): {beta:.4f}" if beta is not None else "Beta (Trend): Ej tillämplig")

    # Använder .get() för att undvika KeyError om 'smoothing_seasonal' inte finns
    gamma = fitted_model.params.get('smoothing_seasonal', None)
    print(f"Gamma (Säsongsvariation): {gamma:.4f}" if gamma is not None else "Gamma (Säsongsvariation): Ej tillämplig")

    # Visualisera den faktiska tidsserien och prognoserna
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_data.index, monthly_data, label='Faktisk')
    plt.plot(forecast.index, forecast, label='Prognos', linestyle='--')
    plt.title('Holt-Winters Tidsanalys (Månadsaggregerad Data)')
    plt.xlabel('Datum')
    plt.ylabel(numeric_column)
    plt.legend()
    plt.show()

# Exempelanvändning:
# Förutsätt att 'dataframe' är din DataFrame med 'Date' som index och 'Weight' eller 'Quantity' som den numeriska kolumnen
# perform_holt_winters_analysis(dataframe, 'Weight', 12)  # Antag 12 för månadsdata med årlig säsongsvariation
def main():
    data = Data('Pappit.xlsx')
    (order_entry_df, order_entry_np), (backlog_df, backlog_np) = data.load_data()

    # Visualisering av pivottabell för 'OrderEntry'
    #visualize_pivot_table(order_entry_df, 'CustomerId', 'ProductCategory', 'Weight', 'sum')

    # Visualisering av korrelationsmatris för 'OrderEntry'
    #visualize_correlation(order_entry_df)
    #visualize_correlation(backlog_df)

    # Visualisering av total vikt och vikt per division över tid
    #visualize_weight_over_time(order_entry_df, 'W')
    #visualize_weight_over_time(backlog_df, 'W')

    # Holt-Winters analys för 'OrderEntry'
    #perform_holt_winters_analysis(order_entry_df,'Weight', 12, 'M')
    print("Number of unique customers:", order_entry_df['CustomerId'].nunique())
if __name__ == '__main__':
    main()
    


