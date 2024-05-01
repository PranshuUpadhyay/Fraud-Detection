import tkinter as tk
from tkinter import messagebox
import pickle  # For loading the trained model

# Function to perform prediction
def predict_transaction():
    # Assuming 'model' is a trained machine learning model
    # Load your model (replace 'model.pkl' with the actual path to your model file)
    with open('FRAUD_DETECTION.py', 'rb') as model_file:
        model = pickle.load(model_file)

    # Gather transaction details from the input fields
    amount = float(amount_entry.get())
    merchant = merchant_entry.get()
    # Other transaction features can be collected similarly

    # Make a prediction using the model
    # Replace this with your actual feature values
    transaction_features = [[amount]]  # Add other features in a similar manner
    prediction = model.predict(transaction_features)

    # Display the prediction result
    if prediction[0] == 1:  # Assuming 1 represents fraud and 0 represents non-fraud
        messagebox.showwarning("Prediction", "This transaction might be fraudulent!")
    else:
        messagebox.showinfo("Prediction", "This transaction seems normal.")

# GUI setup
root = tk.Tk()
root.title("Transaction Fraud Detector")

# Label and entry for transaction amount
amount_label = tk.Label(root, text="Transaction Amount:")
amount_label.pack()
amount_entry = tk.Entry(root)
amount_entry.pack()

# Label and entry for merchant name
merchant_label = tk.Label(root, text="Merchant:")
merchant_label.pack()
merchant_entry = tk.Entry(root)
merchant_entry.pack()

# Button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_transaction)
predict_button.pack()

root.mainloop()