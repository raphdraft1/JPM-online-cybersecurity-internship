import pandas as pd
import matplotlib.pyplot as plt

def visual_1(df):
    def transaction_counts(df):
        return df['type'].value_counts()
    
    def transaction_counts_split_by_fraud(df):
        return df.groupby(by=['type', 'isFraud']).size()

    fig, axs = plt.subplots(2, figsize=(6,10))
    transaction_counts(df).plot(ax=axs[0], kind='bar')
    axs[0].set_title('frequency of transaction types')
    axs[0].set_xlabel('Transaction types')
    axs[0].set_ylabel('Frequency')
    transaction_counts_split_by_fraud(df).plot(ax=axs[1], kind='bar')
    axs[1].set_title('Transaction type frequencies, split by fraud')
    axs[1].set_xlabel('Transaction type, fraud')
    axs[1].set_ylabel('Frequency')
    fig.suptitle('Transaction type')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    for ax in axs:
      for p in ax.patches:
          ax.annotate(p.get_height(), (p.get_x(), p.get_height()))
    plt.show()
    return 'This bar chart shows transaction types split by fraud. It is interesting to note that the only fraudulent transactions are CASH_OUT and TRANSFER. This information will be insightful to management as they can place focus on adding security to these types of transactions. '

df = exercise_0("transactions.csv")

visual_1(df)

def visual_2(df):
    def query(df):
        cash_out_transactions = df[df['type'] == 'CASH_OUT'] 
        return cash_out_transactions[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    
    cash_out_df = query(df)
    plot = query(df).plot.scatter(x='newbalanceOrig',y='newbalanceDest')
    plot.set_title('Origin Account Balance Delta vs Destination Account Balance Delta for Cash Out Transactions')
    plot.set_xlim(left=-1e3, right=1e3)
    plot.set_ylim(bottom=-1e3, top=1e3)
    plt.show()
    return 'Scatter plot showing Origin account balance delta compared to Destination account balance delta for Cash Out transactions.'\
            'When comparing the origin account balance to the destination account balance, it is reassuring that there are no values that are less than 0. '\
            'This means that there are no accounts that are overdrawn, and all accounts meet a minimum balance of 0.0 '
visual_2(df)

def exercise_custom(df):
    summary_stats = df.groupby('type')['amount'].describe()
    return summary_stats
    
def visual_custom(df):
    summary_stats = exercise_custom(df)
    

    summary_stats[['mean', '50%']].plot(kind='bar', figsize=(10, 6))
    plt.title('Distribution of Transaction Amounts by Transaction Type')
    plt.xlabel('Transaction Type')
    plt.ylabel('Transaction Amount')
    plt.xticks(rotation=45)
    plt.legend(['Mean', 'Median'])
    plt.show()

    return "This is a  bar graph that visualizes the distribution of transaction amounts (mean and median) for different transaction types based on the database."


visual_custom(df)