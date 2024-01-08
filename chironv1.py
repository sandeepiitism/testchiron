import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox, Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def train_prophet_model(csv_file):
    df = pd.read_csv(csv_file)
    df.rename(columns={'Time': 'ds', 'Coolant': 'y'}, inplace=True)
    model = Prophet()
    model.fit(df)
    return model

def predict_future_pressure(model, csv_file, days=3):
    df = pd.read_csv(csv_file)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-days:]
    
    return forecast_values

def plot_past_and_future(past_data, future_data):
    plt.figure(figsize=(12, 6))
    plt.plot(past_data['ds'], past_data['Coolant'], label='Past Data', color='blue')
    plt.plot(future_data['ds'], future_data['yhat'], label='Forecasted Data', color='red')
    plt.fill_between(future_data['ds'], future_data['yhat_lower'], future_data['yhat_upper'], color='red', alpha=0.2)
    plt.title('Past and Future Coolant Pressure Forecast')
    plt.xlabel('Date')
    plt.ylabel('Coolant Pressure')
    plt.legend()
    plt.show()

def plot_forecast(future_data):
    # Create a new window for the forecast plot
    forecast_plot_window = Toplevel()
    forecast_plot_window.title('Forecast Plot')

    # Plot past and future data in the new window
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(future_data['ds'], future_data['yhat'], label='Forecasted Data', color='red')
    ax.fill_between(future_data['ds'], future_data['yhat_lower'], future_data['yhat_upper'], color='red', alpha=0.2)
    ax.set_title('Coolant Pressure Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Coolant Pressure')
    ax.legend()


    canvas = FigureCanvasTkAgg(fig, master=forecast_plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chiron_Analytics")
        root.geometry('400x200')
        root.configure(bg='#EFEFEF')
        self.heading_label = Label(root, text="Coolant Prediction", font=("Helvetica", 16), bg='#C0C0C0')
        self.heading_label.pack(side="top", pady=10)

        self.days_label = Label(root, text="Days to predict:")
        self.days_label.pack()

        self.days_entry = Entry(root)
        self.days_entry.pack()

        self.forecast_button = Button(root, text="Forecast", command=self.forecast)
        self.forecast_button.pack()

        self.plot_button = Button(root, text="Plot", command=self.plot_forecast)
        self.plot_button.pack()

    def forecast(self):
        try:
            days_to_predict = int(self.days_entry.get())
            csv_file_path = 'coolant_p.csv'
            prophet_model = train_prophet_model(csv_file_path)
            future_pressure = predict_future_pressure(prophet_model, csv_file_path, days_to_predict)
            
            past_data = pd.read_csv(csv_file_path)
            past_data['ds'] = pd.to_datetime(past_data['Time'])
            past_data = past_data.tail(10)
            
            plot_past_and_future(past_data, future_pressure)
            

            future_pressure.rename(columns={'ds': 'Day', 'yhat': 'Pred_CP', 
                                            'yhat_lower': 'Pred_Low_CP', 
                                            'yhat_upper': 'Predict_High_CP'}, inplace=True)
            
 
            forecast_values_str = future_pressure.to_string(index=False)
            
            messagebox.showinfo("Forecast Results", f"Forecasted values:\n{forecast_values_str}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of days.")

    def plot_forecast(self):
        try:
            days_to_predict = int(self.days_entry.get())
            csv_file_path = 'coolant_p.csv'
            prophet_model = train_prophet_model(csv_file_path)
            future_pressure = predict_future_pressure(prophet_model, csv_file_path, days_to_predict)
            
            plot_forecast(future_pressure)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of days.")

if __name__ == '__main__':
    root = Tk()
    app = GUIApp(root)
    root.mainloop()
